"""
对话 API 服务
======================
把 RAGPipeline 封装成机器人控制层可调用的 REST 接口。

接口：
  GET  /health   健康检查（模型/向量库/熔断器状态），机器人启动时探活
  POST /chat     单轮对话：文本进 → 回复出（机器人主调用）
  POST /reset    清空对话历史（切换游客 / 结束一轮时调用）

用法：
  cd scenic-ai-guide
  python scripts/api_server.py                # 默认 0.0.0.0:8000
  python scripts/api_server.py --port 8000    # 指定端口

设计原则（见 docs/deployment-guide.md）：
  1. 永不返回 5xx 空响应 —— 任何内部异常都转成鲁迅/讲解员口吻的兜底文本，TTS 永远有词可播。
  2. 全局单例 RAGPipeline —— 惰性加载，BGE 模型 / ChromaDB 只加载一次。
  3. 会话隔离 —— 通过 session_id 维护多组 ConversationState，缺省走全局单会话。
  4. 串行化 —— 单一 pipeline + 本地 BGE 推理，用锁串行化请求（demo 场景一个游客同时只问一句）。

依赖：
  - fastapi、uvicorn（新增，需 pip install）
  - 其余复用 rag_pipeline.py 现有依赖
"""

import os

# =========【必须放在所有 transformers 导入之前，与 rag_pipeline.py 对齐】=========
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import sys
import time
import logging
import threading
from typing import Optional, Dict, Literal

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 确保 scripts 目录在 path 中，可 import rag_pipeline / quality_guard 等
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api_server")

# ============================================================
# 全局单例（惰性加载，避免 import 阶段就拉起 torch/BGE）
# ============================================================

_pipeline = None                     # RAGPipeline 全局单例
_pipeline_lock = threading.Lock()    # 保护单例初始化
_pipeline_status = "idle"            # idle | loading | ready | error
_pipeline_error: Optional[str] = None

_request_lock = threading.Lock()     # 串行化 ask()，防止并发踩同一 pipeline.state / BGE

_sessions: Dict[str, object] = {}    # session_id -> ConversationState
_sessions_lock = threading.Lock()

# 兜底话术（与 quality_guard.get_fallback 的 FALLBACK_API_ERROR 保持一致）
GENERIC_FALLBACK = "这大约是什么缘故呢——我此刻竟想不起来了。你先问些别的罢。"

# 意图 → 中文标签（与 gradio_app.py 对齐）
INTENT_LABELS = {
    "narrator":  "🏛️ 讲解员模式",
    "luxun":     "🎭 鲁迅数字人",
    "ambiguous": "🎭 鲁迅数字人（自动）",
    "reject_time":       "⚠️ 时间越界·已拦截",
    "reject_irrelevant": "🚫 无关内容·已拒绝",
}


def get_pipeline():
    """惰性获取 RAGPipeline 单例；首次调用会加载 BGE + ChromaDB（约 1 分钟，实测 ~77s）。"""
    global _pipeline, _pipeline_status, _pipeline_error
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline_status = "loading"
                logger.info("首次加载 RAG 全链路（BGE 模型 + ChromaDB + LLM 客户端）...")
                try:
                    from rag_pipeline import RAGPipeline
                    _pipeline = RAGPipeline()
                    _pipeline_status = "ready"
                    logger.info("RAG 全链路就绪。")
                except Exception as e:
                    _pipeline_status = "error"
                    _pipeline_error = str(e)
                    logger.exception("RAG 全链路加载失败")
                    raise
    return _pipeline


def _get_session_state(session_id: Optional[str]):
    """按 session_id 获取（或创建）会话状态；无 session_id 返回 None（用全局默认状态）。"""
    if not session_id:
        return None
    from rag_pipeline import ConversationState
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = ConversationState()
        return _sessions[session_id]


def _build_debug(pipeline) -> dict:
    """从 pipeline.last_debug 提取调试信息（仅在 include_debug=True 时返回）。"""
    d = pipeline.last_debug or {}
    return {
        "confidence": d.get("confidence", 0),
        "reason": d.get("reason", ""),
        "matched": d.get("matched", []),
        "time_boundary": d.get("time_boundary", False),
        "rewrite_used": d.get("rewrite_used", False),
        "rewrite_concepts": d.get("rewrite_concepts", []),
        "rewrite_queries": d.get("rewrite_queries", []),
        "chunks": d.get("chunks", []),
        "guard_details": d.get("guard_details", {}),
        "circuit_status": d.get("circuit_status"),
    }


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="鲁迅数字人 · 对话大脑 API",
    description="机器人语音层对接接口：ASR 文本进 → 回复文本出。",
    version="0.1.0",
)

# 放开 CORS（机器人控制层若为 Web 应用 / 本地调试均可跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 数据模型
# ============================================================

class ChatRequest(BaseModel):
    text: str = Field("", description="用户问题文本（ASR 识别结果）")
    session_id: Optional[str] = Field(None, description="会话 ID，多轮对话隔离用；缺省走全局单会话")
    force_mode: Optional[Literal["narrator", "luxun"]] = Field(
        None, description="强制模式；缺省自动意图识别"
    )
    include_debug: bool = Field(False, description="是否返回调试信息（检索片段/守卫详情等）")


class ChatResponse(BaseModel):
    success: bool = Field(..., description="是否成功（内部兜底也算成功，只要返回了可播报文本）")
    reply: str = Field(..., description="回复文本，可直接交给 TTS")
    intent: str = Field("", description="识别出的意图（narrator/luxun/ambiguous/reject_*）")
    mode: str = Field("", description="模式中文标签（给人看，机器人可忽略）")
    guard_status: str = Field("", description="质量守卫判定：PASS/WARN/AMEND/FALLBACK/N/A")
    latency_ms: int = Field(0, description="本问耗时（毫秒）")
    error: Optional[str] = Field(None, description="仅当 success=false 时给出错误描述")
    debug: Optional[dict] = Field(None, description="仅当 include_debug=true 时返回")


# ============================================================
# 接口
# ============================================================

@app.get("/")
def root():
    return {"service": "鲁迅数字人 · 对话大脑", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    """健康检查：探测 pipeline 是否加载成功 + 熔断器状态。会触发惰性加载（可当作预热）。"""
    try:
        get_pipeline()
        ok = True
        status = "ok"
        error = None
    except Exception as e:
        ok = False
        status = "error"
        error = str(e)

    circuit = None
    if _pipeline is not None:
        circuit = _pipeline.last_debug.get("circuit_status") or _pipeline.llm.circuit_status

    return {
        "status": status,
        "model_loaded": ok,
        "error": error,
        "circuit": circuit,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """单轮对话。永不返回 5xx 空响应——异常一律转兜底文本。"""
    t0 = time.time()

    # 1. 加载 pipeline（失败 → 兜底，不抛 5xx）
    try:
        pipeline = get_pipeline()
    except Exception as e:
        return ChatResponse(
            success=False,
            reply=GENERIC_FALLBACK,
            guard_status="ERROR",
            latency_ms=int((time.time() - t0) * 1000),
            error=f"pipeline 加载失败: {e}",
        )

    # 2. 会话状态切换（有 session_id 则换用该会话的状态）
    session_state = _get_session_state(req.session_id)
    saved_state = pipeline.state
    if session_state is not None:
        pipeline.state = session_state

    # 3. 调用管线（串行化）
    try:
        with _request_lock:
            reply = pipeline.ask(req.text, force_mode=req.force_mode)
            intent = pipeline.state.current_intent
            guard_status = pipeline.last_debug.get("guard_status") or "N/A"
            latency_ms = pipeline.last_debug.get("elapsed_ms", 0)
            debug = _build_debug(pipeline) if req.include_debug else None
    except Exception as e:
        # 意外异常（非 LLM 失败，而是意图/检索/检查器 bug）→ 兜底
        logger.exception("ask() 发生未捕获异常")
        reply = GENERIC_FALLBACK
        intent = pipeline.state.current_intent if pipeline.state else ""
        guard_status = "ERROR"
        latency_ms = int((time.time() - t0) * 1000)
        debug = None
    finally:
        if session_state is not None:
            pipeline.state = saved_state

    return ChatResponse(
        success=True,
        reply=reply,
        intent=intent,
        mode=INTENT_LABELS.get(intent, intent),
        guard_status=guard_status,
        latency_ms=latency_ms,
        debug=debug,
    )


@app.post("/reset")
def reset(session_id: Optional[str] = Query(None, description="要重置的会话；缺省重置全局")):
    """清空对话历史。按 session_id 清单会话，或缺省清全局。"""
    if session_id:
        with _sessions_lock:
            existed = _sessions.pop(session_id, None) is not None
        return {"success": True, "reset": session_id, "existed": existed}

    # 全局重置
    if _pipeline is not None:
        try:
            from rag_pipeline import ConversationState
            with _request_lock:
                _pipeline.state = ConversationState()
        except Exception:
            logger.exception("全局重置失败")
    return {"success": True, "reset": "global"}


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="鲁迅数字人对话 API 服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定地址（0.0.0.0 允许局域网/机器人访问）")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    args = parser.parse_args()

    print(f"对话 API 服务启动 → http://{args.host}:{args.port}  (接口文档: /docs)")
    uvicorn.run(app, host=args.host, port=args.port)
