"""
RAG 全链路
==================
意图分类 → 越界改写 → 知识检索 → Prompt 拼接 → LLM 生成 → 多轮对话

越界方案（5层防护）：
  Layer 1: 意图分类器拦截 — reject_time 高置信度直接拒绝
  Layer 2: 查询改写拆解 — 中置信度时，将现代概念映射为时间安全子查询
  Layer 3: 多轮检索合并 — 多个子查询分别检索，合并去重
  Layer 4: System Prompt 时间边界 — 鲁迅只知道1936年前的事
  Layer 5: QualityGuard 回答后卫 — 幻觉检测 + 一致性校验 + 重试/兜底

架构：
  用户输入 → intent_classifier (5分类)
           → [time_aware + 中置信度] → query_rewriter (概念映射+改写拆解)
           → 按意图选择检索域 + 多子查询检索合并
           → ChromaDB 语义检索
           → 拼接 System Prompt + Context + User Query
           → LLM 生成
           → 输出 + 对话历史记录

用法：
  python scripts/rag_pipeline.py          # 交互模式
  python scripts/rag_pipeline.py --once "鲁迅的原名是什么？"  # 单次问答

依赖：
  - scripts/query.py              (ChromaDB + BGE 检索)
  - scripts/llm_client.py         (LLM API 统一调用)
  - scripts/intent_classifier.py  (意图分类)
  - scripts/query_rewriter.py     (越界查询改写)
  - prompts/luxun_digital_human_v3.md  (数字人 Prompt)
  - prompts/venue_narrator_v2.md       (讲解员 Prompt)
"""

import os
# =========【必须放在所有transformers导入之前！】=========
# 适配transformers5.x，消除各类警告、路径识别bug
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# 国内本地环境走 hf-mirror 镜像加速；云端部署改用官方源
# （HuggingFace Spaces 有 SPACE_ID，Render 有 RENDER），否则第三方镜像可能
# 限流/不可达，导致 BGE 模型在云端无法下载。
if not os.environ.get("SPACE_ID") and not os.environ.get("RENDER"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import json
import time
import logging
import argparse
from typing import Optional, List, Dict
from dataclasses import dataclass, field

# 屏蔽transformers冗余日志
logging.getLogger("transformers").setLevel(logging.ERROR)

# 确保 scripts 目录在 path 中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import chromadb
from transformers import AutoTokenizer, AutoModel
import torch

from intent_classifier import classify
from llm_client import LLMClient, get_circuit_status
from query_rewriter import QueryRewriter, RewriteResult
from quality_guard import QualityGuard, GuardResult, get_fallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag_pipeline")

# ============================================================
# 模型配置 — 与 ingest.py/query.py 完全对齐
# ============================================================
LOCAL_MODEL_PATH = os.path.join(SCRIPT_DIR, "bge-small-zh-v1.5")
HF_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 优先本地路径（Transformers老版本环境），不存在则用 HF 名称（新版环境）
if os.path.isdir(LOCAL_MODEL_PATH):
    MODEL_PATH = LOCAL_MODEL_PATH
else:
    MODEL_PATH = HF_MODEL_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(device)
model.eval()


# ============================================================
# 嵌入缓存 — 对重复查询复用向量，减少 BGE 推理耗时
# ============================================================
_embedding_cache: dict = {}
_EMBED_CACHE_MAX = 256

def get_embedding(text: str) -> list:
    """查询向量化 — 与 ingest.py 保持 Mean Pooling 一致，带 LRU 缓存"""
    # 用 hash 做 key，避免长文本占用缓存内存
    key = text[:200]  # 前200字符相同即命中
    if key in _embedding_cache:
        return _embedding_cache[key]

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = model(**inputs)
    vec = out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()

    # LRU 淘汰：超过上限时删除最早的一半
    if len(_embedding_cache) >= _EMBED_CACHE_MAX:
        keys_to_del = list(_embedding_cache.keys())[:_EMBED_CACHE_MAX // 2]
        for k in keys_to_del:
            del _embedding_cache[k]
    _embedding_cache[key] = vec
    return vec


def _content_fingerprint(text: str) -> str:
    """提取文本前80字符作为内容指纹，用于去重"""
    return text.strip()[:80]

# ============================================================
# 业务配置
# ============================================================
CHROMA_STORE = os.path.join(ROOT_DIR, "chroma_db")
COLLECTION_NAME = "luxun_know_base"
TOP_K = 5
MAX_HISTORY_ROUNDS = 5     # 多轮对话保留最近 N 轮

# Prompt 文件
PROMPT_LUXUN_FILE = os.path.join(ROOT_DIR, "prompts", "luxun_digital_human_v3.md")
PROMPT_NARRATOR_FILE = os.path.join(ROOT_DIR, "prompts", "venue_narrator_v2.md")

# 知识域 → 意图映射
DOMAIN_FILTER_MAP = {
    "narrator":   ["venue", "bio"],
    "luxun":      ["work", "quote", "persona", "bio"],
    "ambiguous":  ["venue", "bio", "work", "quote", "persona"],
    # reject_* 不检索
}

# 越界回复模板（不检索，直接用鲁迅口吻回答）
REJECT_TIME_RESPONSE_HINTS = {
    "现代科技": "这大约是什么新奇的东西罢。我生于光绪七年，殁于民国二十五年，怕是未曾见过。大抵是我所不能知道的事了。",
    "现代生活": "至于你说的情况，我想，那是属于另一个时代的事了。我的手头所知，尽在1936年之前。",
    "假设活在今天": "倘要我设想后来的世界——我因在1936年便已闭眼，未曾见过。但我想，无论什么时代，青年总该是有希望的。",
}

REJECT_IRRELEVANT_RESPONSE = (
    "这个问题与鲁迅纪念馆或鲁迅先生本人无关。"
    "你可以问我关于纪念馆的参观信息，或者向鲁迅先生提问关于他的作品与思想。"
)


# ============================================================
# 检索器【不再内部重复加载模型，复用全局tokenizer/model/get_embedding】
# ============================================================

class Retriever:
    """封装 BGE 模型 + ChromaDB 检索 + 领域过滤"""

    def __init__(self):
        logger.info("ChromaDB 已连接, collection=%s", COLLECTION_NAME)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_STORE)
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)

    def search(
        self, query: str, domain_filter: Optional[List[str]] = None, top_k: int = TOP_K
    ) -> list[dict]:
        """
        语义检索 + 可选的 type 领域过滤 + 内容去重 + 查询扩展。

        Args:
            query:         用户问题文本
            domain_filter: 限定检索的 type 列表，如 ["venue", "bio"]；None 表示全量
            top_k:         返回条数

        Returns:
            [{"content": str, "title": str, "type": str, "source": str,
              "year": int, "distance": float}, ...]
        """
        # 检索比最终需要更多的结果（为去重留余量）
        fetch_k = max(top_k * 2, 10)
        vec = get_embedding(query)

        where_clause = None
        if domain_filter:
            where_clause = {"type": {"$in": domain_filter}}

        res = self.collection.query(
            query_embeddings=[vec],
            n_results=fetch_k,
            where=where_clause,
        )

        # 结构化 + 内容去重
        results = []
        seen_fingerprints = set()
        seen_ids = set()

        if res["ids"] and res["ids"][0]:
            for i in range(len(res["ids"][0])):
                cid = res["ids"][0][i]
                content = res["documents"][0][i] if res["documents"] else ""

                # ID 去重
                if cid and cid in seen_ids:
                    continue
                if cid:
                    seen_ids.add(cid)

                # 内容去重：相同文本前缀的不重复收录
                fp = _content_fingerprint(content)
                if fp in seen_fingerprints:
                    continue
                seen_fingerprints.add(fp)

                meta = res["metadatas"][0][i] if res["metadatas"] else {}
                results.append({
                    "chunk_id": cid,
                    "content": content,
                    "type": meta.get("type", "unknown"),
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "year": meta.get("year", 0),
                    "character_voice": meta.get("character_voice", False),
                    "venue_relevant": meta.get("venue_relevant", False),
                    "distance": res["distances"][0][i] if res["distances"] else 0.0,
                })

                # 收到足够多去重结果即停止
                if len(results) >= top_k:
                    break

        # 查询扩展：如果去重后结果太少，用问题中的关键词单独检索补充
        if len(results) < top_k and len(query) > 5:
            extra = self._expand_search(query, domain_filter, top_k - len(results), seen_ids, seen_fingerprints)
            results.extend(extra)

        return results

    def _expand_search(
        self, query: str, domain_filter: Optional[List[str]], need: int,
        seen_ids: set, seen_fingerprints: set,
    ) -> list[dict]:
        """
        查询扩展：提取问题中的关键词做补充检索，提高结果丰富度。
        用于首次检索结果太少或太单一的情况。
        """
        import jieba
        import jieba.posseg  # 显式加载词性标注子模块
        # 提取长度≥2的名词/动词作为关键词
        keywords = []
        for w, flag in jieba.posseg.cut(query):
            if len(w) >= 2 and flag in ("n", "v", "nr", "ns", "nt", "nz"):
                if w not in ("鲁迅", "先生", "什么", "怎么", "哪些", "为什么", "请问", "你可以"):
                    keywords.append(w)

        # 最多取3个关键词
        keywords = list(dict.fromkeys(keywords))[:3]  # 去重保序
        if not keywords:
            return []

        extra_results = []
        for kw in keywords:
            if len(extra_results) >= need:
                break
            kw_results = self.search(kw, domain_filter=domain_filter, top_k=3)
            for r in kw_results:
                cid = r.get("chunk_id", "")
                fp = _content_fingerprint(r.get("content", ""))
                if cid in seen_ids or fp in seen_fingerprints:
                    continue
                if cid:
                    seen_ids.add(cid)
                seen_fingerprints.add(fp)
                extra_results.append(r)
                if len(extra_results) >= need:
                    break

        return extra_results


# ============================================================
# Prompt 管理
# ============================================================

class PromptManager:
    """加载 + 缓存双模式 System Prompt"""

    def __init__(self):
        self._luxun_prompt: Optional[str] = None
        self._narrator_prompt: Optional[str] = None

    @property
    def luxun(self) -> str:
        if self._luxun_prompt is None:
            self._luxun_prompt = self._load(PROMPT_LUXUN_FILE)
        return self._luxun_prompt

    @property
    def narrator(self) -> str:
        if self._narrator_prompt is None:
            self._narrator_prompt = self._load(PROMPT_NARRATOR_FILE)
        return self._narrator_prompt

    @staticmethod
    def _load(path: str) -> str:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        logger.warning("Prompt 文件不存在: %s，使用内置默认", path)
        return "你是专业的讲解员/鲁迅数字人，请根据参考资料回答问题。"


# ============================================================
# 上下文格式化
# ============================================================

def format_context(chunks: list[dict], max_chunks: int = 6) -> str:

    if not chunks:
        return "（未检索到相关知识）"

    lines = []
    for i, c in enumerate(chunks[:max_chunks], 1):
        meta_str = f"type:{c['type']} | {c['title']}"
        if c.get("source"):
            meta_str += f" | 来源:{c['source']}"
        if c.get("year"):
            meta_str += f" ({c['year']})"

        lines.append(f"[{i}] {meta_str}")
        lines.append(f"    {c['content']}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# 对话状态
# ============================================================

@dataclass
class ConversationState:
    """多轮对话状态管理"""
    history: List[Dict[str, str]] = field(default_factory=list)
    current_intent: str = "ambiguous"
    turn_count: int = 0

    def add(self, user: str, assistant: str):
        # 强类型转换：禁止 dict/对象 存入 content，避免 LLM API 400
        user = str(user)
        assistant = str(assistant)
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": assistant})
        self.turn_count += 1

        # 只保留最近 MAX_HISTORY_ROUNDS 轮
        max_messages = MAX_HISTORY_ROUNDS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_recent_qa(self) -> Optional[str]:
        if len(self.history) >= 2:
            # 安全防御：用 .get 规避 KeyError，强转 str 规避 None/切片报错
            last_user = self.history[-2].get("content", "")
            last_assistant = self.history[-1].get("content", "")
            last_user = str(last_user) if last_user is not None else ""
            last_assistant = str(last_assistant) if last_assistant is not None else ""
            return f"上文: 问「{last_user[:80]}」答「{last_assistant[:60]}」"
        return None

    def to_messages(self, system_prompt: str) -> list[dict]:
        """转为 LLM API 的 messages 格式"""
        return [{"role": "system", "content": system_prompt}] + self.history


# ============================================================
# 主管线
# ============================================================

class RAGPipeline:
    """RAG 全链路：意图分类 → 检索 → Prompt → LLM → 对话状态"""

    def __init__(self):
        logger.info("=" * 50)
        logger.info("初始化 RAG 全链路...")
        logger.info("=" * 50)

        self.retriever = Retriever()
        self.prompts = PromptManager()
        self.llm = LLMClient()
        self.rewriter = QueryRewriter(llm_client=self.llm)
        self.guard = QualityGuard()
        self.state = ConversationState()
        self.last_debug = {}  # 调试信息，每次 ask() 后更新

        logger.info("RAG 全链路就绪（含越界改写层）。")

    # ---- 检索策略 ----

    def _retrieve(self, user_query: str, intent: str) -> list[dict]:
        """根据意图执行检索"""
        domains = DOMAIN_FILTER_MAP.get(intent)

        # 多轮对话：用最近一轮 QA 做 query expansion
        recent = self.state.get_recent_qa()
        expanded_query = user_query
        if recent:
            expanded_query = f"{recent}\n当前问题: {user_query}"

        if domains:
            return self.retriever.search(expanded_query, domain_filter=domains)
        return []

    def _retrieve_multi(
        self, sub_queries: List[str], intent: str, top_k_per_query: int = 3
    ) -> list[dict]:
        """
        多子查询检索 + 合并去重。
        用于越界改写后的多轮检索（Layer 3）。

        Args:
            sub_queries:      改写后的多个子查询
            intent:           意图（决定检索域）
            top_k_per_query:  每个子查询的返回条数

        Returns:
            合并去重后的检索结果列表（按 distance 排序）
        """
        seen_ids = set()
        merged = []

        for sq in sub_queries:
            chunks = self._retrieve(sq, intent)
            for c in chunks:
                cid = c.get("chunk_id", "")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    merged.append(c)
                elif not cid:
                    # 无 chunk_id 的兜底去重：用 content 前 60 字符
                    key = c.get("content", "")[:60]
                    if key not in seen_ids:
                        seen_ids.add(key)
                        merged.append(c)

        # 按 distance 升序（越相似越靠前）
        merged.sort(key=lambda x: x.get("distance", 999.0))

        logger.info(
            "多子查询检索: %d 子查询 → %d 条去重结果 (top_k=%d)",
            len(sub_queries), len(merged), top_k_per_query,
        )

        return merged[:TOP_K * 2]  # 最多返回 2 倍 Top‑K

    # ---- 越界改写入口 (Layer 2) ----

    def _try_rewrite(
        self, user_input: str, intent_result: dict, verbose: bool = False
    ) -> Optional[RewriteResult]:
        """
        尝试对时间越界问题进行改写。

        触发条件（任一满足即触发）：
          (a) reject_time → 始终先尝试改写（挽留优先，改写失败才拒绝）
          (b) time_aware + time_warning（低置信越界已降级）
          (c) 改写器自身检测到现代概念（补充 Layer 1 未覆盖的概念）

        Returns:
            RewriteResult 如果改写成功，None 如果不需要改写
        """
        intent = intent_result["intent"]
        confidence = intent_result.get("confidence", 0)

        # 判断是否需要改写
        need_rewrite = False

        if intent == "reject_time":
            need_rewrite = True  # 所有越界都先尝试改写挽留
        elif intent_result.get("time_aware") and intent_result.get("time_warning"):
            need_rewrite = True

        # 补充：即使 Layer 1 未检测到，也让改写器扫描一次
        # 因为 Layer 1 的词表有限，改写器有更完善的概念映射表
        if not need_rewrite:
            # 轻量扫描（不调用 LLM）
            probe = self.rewriter.rewrite(
                user_input, intent_result=intent_result, use_llm=False
            )
            if probe.modern_concepts:
                need_rewrite = True
                if verbose:
                    logger.info(
                        "Layer 2: 改写器检测到 Layer 1 未覆盖的现代概念: %s",
                        probe.modern_concepts,
                    )

        if not need_rewrite:
            return None

        if verbose:
            logger.info(
                "Layer 2: 尝试查询改写 (intent=%s, conf=%.2f, time_aware=%s)",
                intent, confidence, intent_result.get("time_aware"),
            )

        # 调用改写器（启用 LLM 辅助复杂改写）
        rewrite_result = self.rewriter.rewrite(
            user_input,
            intent_result=intent_result,
            use_llm=True,
        )

        if verbose:
            logger.info(
                "改写结果: can_rewrite=%s, %d 子查询, concepts=%s",
                rewrite_result.can_rewrite,
                len(rewrite_result.sub_queries),
                rewrite_result.modern_concepts,
            )

        return rewrite_result

    # ---- 单轮问答 ----

    def ask(self, user_input: str, verbose: bool = False, force_mode: str = None, return_retrieval: bool = False):
        """
        处理一轮对话。

        越界方案（4层）：
          Layer 1: intent_classifier 拦截高置信越界
          Layer 2: query_rewriter 改写中置信越界
          Layer 3: 多子查询检索合并
          Layer 4: System Prompt 时间边界

        Args:
            user_input: 用户原始输入
            verbose:    是否打印调试信息
            force_mode: 强制模式 None/ "narrator"/"luxun"，不为None时跳过自动意图识别
            return_retrieval: True时返回 (reply, chunks)；False只返回reply字符串

        Returns:
            LLM 生成的回复文本；return_retrieval=True 返回元组(回复文本,检索片段列表)
        """
        user_input = user_input.strip()
        if not user_input:
            if return_retrieval:
                return "（请输入你的问题）", []
            return "（请输入你的问题）"

        t_start = time.time()
        self.last_debug = {
            "intent": "", "confidence": 0, "reason": "", "matched": [],
            "time_boundary": False, "time_warning": "",
            "rewrite_used": False, "rewrite_concepts": [], "rewrite_queries": [],
            "rewrite_fallback": "",
            "chunks": [], "chunk_count": 0,
            "elapsed_ms": 0, "reply_len": 0, "error": None,
            "guard_status": "", "guard_details": {},
        }

        # ── Step 1: 意图分类 (Layer 1) ──
        if force_mode is not None:
            intent = force_mode
            intent_result = {"intent": force_mode, "confidence": 1.0, "reason": "manual_force"}
            self.state.current_intent = intent
        else:
            intent_result = classify(user_input)
            intent = intent_result["intent"]
            self.state.current_intent = intent

        self.last_debug.update({
            "intent": intent,
            "confidence": intent_result.get("confidence", 0),
            "reason": intent_result.get("reason", ""),
            "matched": intent_result.get("matched", [])[:5],
            "time_boundary": intent_result.get("time_aware", False),
            "time_warning": intent_result.get("time_warning", ""),
        })

        if verbose:
            logger.info("意图: %s (conf=%.2f) reason=%s",
                        intent, intent_result["confidence"], intent_result["reason"])
            if intent_result.get("matched"):
                logger.info("  命中: %s", intent_result["matched"][:3])

        # ── Step 2: 按意图路由 ──

        # 2a. 无关/恶意 → 直接拒绝
        if intent == "reject_irrelevant":
            reply = REJECT_IRRELEVANT_RESPONSE
            self.last_debug["elapsed_ms"] = int((time.time() - t_start) * 1000)
            self.last_debug["reply_len"] = len(reply)
            self.state.add(user_input, reply)
            if return_retrieval:
                return reply, []
            return reply

        # 2b. 越界改写尝试 (Layer 2) — 所有 reject_time / time_aware 都先尝试改写
        #     如果改写成功，检索改写后的子查询并生成回复
        #     如果改写失败，用鲁迅口吻拒绝
        rewrite_result = self._try_rewrite(user_input, intent_result, verbose=verbose)

        rewrite_notice = ""  # 用于告知 LLM 经过了改写
        if rewrite_result and rewrite_result.can_rewrite and rewrite_result.sub_queries:
            # ── Layer 3: 多子查询检索合并 ──
            search_intent = intent if intent not in ("reject_time",) else "ambiguous"
            chunks = self._retrieve_multi(
                rewrite_result.sub_queries, search_intent
            )

            self.last_debug.update({
                "rewrite_used": True,
                "rewrite_concepts": rewrite_result.modern_concepts[:5] if rewrite_result.modern_concepts else [],
                "rewrite_queries": rewrite_result.sub_queries[:5],
            })

            # 构建改写说明
            if rewrite_result.modern_concepts:
                concepts_str = "、".join(rewrite_result.modern_concepts[:3])
                subs_str = "；".join(rewrite_result.sub_queries[:3])
                rewrite_notice = (
                    "\n\n[系统提醒：用户问题涉及「" + concepts_str + "」等概念，"
                    "这些在1936年后才出现。已将问题改写为：「" + subs_str + "」。"
                    "请基于参考资料回答这些改写后的问题，并在回答开头用鲁迅口吻简要说明"
                    "你只了解1936年前的事物。]"
                )
        elif rewrite_result and not rewrite_result.can_rewrite:
            # 改写失败 → 用越界话术回复
            reply = rewrite_result.fallback_message or (
                "这大约是什么新奇的东西罢。"
                "我生于光绪七年，殁于民国二十五年，怕是未曾见过。"
                "大抵是我所不能知道的事了。"
            )
            self.last_debug.update({
                "rewrite_used": True,
                "rewrite_fallback": "改写失败，使用越界话术",
            })
            self.last_debug["elapsed_ms"] = int((time.time() - t_start) * 1000)
            self.last_debug["reply_len"] = len(reply)
            self.state.add(user_input, reply)
            if return_retrieval:
                return reply, []
            return reply
        else:
            # 正常/模糊 → 直接检索
            chunks = self._retrieve(user_input, intent)

        if verbose:
            logger.info("检索到 %d 条 (intent=%s)", len(chunks), intent)
            for i, c in enumerate(chunks[:3]):
                logger.info("  [%d] type:%s | %s | dist:%.2f",
                            i + 1, c["type"], c["title"][:40], c["distance"])

        # 存储检索结果摘要供调试面板使用
        chunk_summaries = []
        for c in chunks[:5]:
            chunk_summaries.append({
                "title": (c.get("title") or "")[:60],
                "type": c.get("type", ""),
                "distance": round(c.get("distance", 0), 3),
                "snippet": (c.get("content") or "")[:100],
            })
        self.last_debug["chunks"] = chunk_summaries
        self.last_debug["chunk_count"] = len(chunks)

        # ── Layer 4: System Prompt 时间边界 ──
        if intent == "narrator":
            system_prompt = self.prompts.narrator
        else:
            system_prompt = self.prompts.luxun

        # 时间边界提示
        time_boundary_note = ""
        if intent_result.get("time_aware") and intent_result.get("time_warning"):
            time_boundary_note = (
                f"\n\n[系统提醒：用户问题涉及 {intent_result['time_warning']}，"
                f"这些概念在1936年后才出现。请按时间边界规则处理，不要编造。]"
            )

        # 合并改写说明和时间边界
        if rewrite_notice:
            system_prompt = system_prompt + rewrite_notice
        elif time_boundary_note:
            system_prompt = system_prompt + time_boundary_note

        # 格式化上下文
        context_str = format_context(chunks)

        # ── Step 3: 调用 LLM ──
        def _call_llm(temp_modifier: float = 0.0):
            """内部函数：用指定参数调用 LLM"""
            temp = max(0.05, 0.3 + temp_modifier)
            if len(self.state.history) >= 2:
                messages = self.state.to_messages(system_prompt)
                user_content = (
                    f"【参考资料】\n{context_str}\n\n"
                    f"【用户问题】\n{user_input}"
                ) if context_str else user_input
                messages.append({"role": "user", "content": user_content})
                res = self.llm.call(messages, temperature=temp)
                # call() 返回结构化 dict；失败时抛出，交由上层兜底逻辑处理
                if not res.get("success"):
                    raise RuntimeError(res.get("error") or "LLM 调用失败")
                return res.get("content", "")
            else:
                return self.llm.chat(
                    system_prompt=system_prompt,
                    context=context_str,
                    user_query=user_input,
                    temperature=temp,
                )

        reply = None
        guard_error = None
        try:
            reply = _call_llm()
        except Exception as e:
            logger.error("LLM 调用失败: %s", e)
            guard_error = str(e)
            reply = get_fallback(intent, "api_error")
            self.last_debug["error"] = guard_error
            self.last_debug["guard_status"] = "FAIL"

        # ── Step 3.5: QualityGuard 回答后校验 (Layer 5) ──
        if not guard_error and reply:
            guard_report = self.guard.evaluate(
                response=reply,
                context=chunks,
                intent=intent,
                query=user_input,
            )

            self.last_debug["guard_status"] = guard_report.result.value
            self.last_debug["guard_details"] = guard_report.details

            if guard_report.result == GuardResult.RETRY:
                # 重试一次（降低 temperature 以获得更保守的回答）
                logger.warning(
                    "QualityGuard RETRY: %s — 正在重试 (temperature↓)",
                    guard_report.retry_reason,
                )
                self.last_debug["guard_details"]["retry_reason"] = guard_report.retry_reason

                try:
                    retry_reply = _call_llm(temp_modifier=-0.15)
                    retry_report = self.guard.evaluate(
                        response=retry_reply,
                        context=chunks,
                        intent=intent,
                        query=user_input,
                    )

                    if retry_report.result == GuardResult.RETRY:
                        # 二次不合格 → 兜底话术
                        logger.warning("QualityGuard 重试仍不合格 → 兜底话术")
                        reply = get_fallback(intent, "guard_fail")
                        self.last_debug["guard_status"] = "FALLBACK"
                        self.last_debug["guard_details"]["final_action"] = "fallback"
                    elif retry_report.result == GuardResult.AMEND:
                        reply = self.guard.amend(retry_reply, retry_report.amend_text)
                        self.last_debug["guard_status"] = "AMEND"
                        self.last_debug["guard_details"]["final_action"] = "amend_after_retry"
                    else:
                        reply = retry_reply
                        self.last_debug["guard_status"] = "PASS"
                        self.last_debug["guard_details"]["final_action"] = "pass_after_retry"
                except Exception as e:
                    logger.error("QualityGuard 重试失败: %s", e)
                    reply = get_fallback(intent, "api_error")
                    self.last_debug["guard_status"] = "FALLBACK"

            elif guard_report.result == GuardResult.AMEND:
                reply = self.guard.amend(reply, guard_report.amend_text)
                logger.info("QualityGuard AMEND: 已追加边界声明")

        # ── Step 4: 保存对话历史 ──
        self.state.add(user_input, reply)

        # 追加熔断器状态，供前端调试面板读取
        self.last_debug["circuit_status"] = get_circuit_status()

        self.last_debug["elapsed_ms"] = int((time.time() - t_start) * 1000)
        self.last_debug["reply_len"] = len(reply)

        # 返回分支
        if return_retrieval:
            return reply, chunks
        return reply
    # ---- 越界处理 ----

    def _handle_reject_time(self, user_input: str, intent_result: dict) -> str:
        """处理时间越界：用鲁迅口吻表达困惑，不检索知识库"""
        # 根据匹配到的越界类型选择参考话术
        hint = REJECT_TIME_RESPONSE_HINTS.get("现代科技",
            "这大约是什么新奇的东西罢。我生于光绪七年，殁于民国二十五年，怕是未曾见过。大抵是我所不能知道的事了。")

        # 构建越界 Prompt
        system_prompt = self.prompts.luxun + (
            "\n\n特别注意：用户问题涉及1936年后的概念或事物。"
            "请用鲁迅的口吻真诚地表达困惑，简短回应（2‑3句话即可）。"
        )

        try:
            reply = self.llm.chat(
                system_prompt=system_prompt,
                context="（无相关知识——用户问题涉及1936年后的事物，鲁迅不可能知晓）",
                user_query=user_input,
            )
        except Exception as e:
            logger.error("LLM 调用失败: %s", e)
            reply = hint

        return reply

    # ---- 交互模式 ----

    def run_interactive(self):
        """命令行交互式对话"""
        print()
        print("╔" + "═" * 58 + "╗")
        print("║  鲁迅数字人智能对话系统                                  ║")
        print("║  讲解员模式 | 鲁迅口吻 | 自动意图识别                    ║")
        print("║  输入 /verbose 切换调试模式  |  /reset 重置对话  |  exit ║")
        print("╚" + "═" * 58 + "╝")
        print()

        verbose = False

        while True:
            try:
                prefix = "🏛️ " if self.state.current_intent == "narrator" else "🎭 "
                user_input = input(prefix + "你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break

            if not user_input:
                continue

            # 命令
            if user_input.lower() == "exit":
                print("再见。")
                break
            if user_input == "/verbose":
                verbose = not verbose
                print(f"调试模式: {'开' if verbose else '关'}")
                continue
            if user_input == "/reset":
                self.state = ConversationState()
                print("对话历史已重置。")
                continue

            # 生成回复
            start = time.time()
            reply = self.ask(user_input, verbose=verbose)
            elapsed = time.time() - start

            # 打印回复
            intent_label = {
                "narrator": "讲解模式",
                "luxun": "鲁迅",
                "ambiguous": "鲁迅（自动）",
                "reject_time": "⚠️ 越界拦截",
                "reject_irrelevant": "🚫 拒绝",
            }.get(self.state.current_intent, self.state.current_intent)

            print(f"\n  [{intent_label}] {reply}")
            if verbose:
                print(f"  (耗时 {elapsed:.1f}s)")
            print()


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="鲁迅数字人 RAG 全链路对话")
    parser.add_argument("--once", "-o", type=str, default=None,
                        help="单次问答（非交互模式）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印调试信息")
    args = parser.parse_args()

    pipeline = RAGPipeline()

    if args.once:
        reply = pipeline.ask(args.once, verbose=args.verbose)
        print(reply)
    else:
        pipeline.run_interactive()


if __name__ == "__main__":
    main()