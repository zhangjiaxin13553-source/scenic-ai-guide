"""
RAG 全链路
==================
意图分类 → 越界改写 → 知识检索 → Prompt 拼接 → LLM 生成 → 多轮对话

越界方案（4层防护）：
  Layer 1: 意图分类器拦截 — reject_time 高置信度直接拒绝
  Layer 2: 查询改写拆解 — 中置信度时，将现代概念映射为时间安全子查询
  Layer 3: 多轮检索合并 — 多个子查询分别检索，合并去重
  Layer 4: System Prompt 时间边界 — 鲁迅只知道1936年前的事

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
  - prompts/luxun_digital_human_v2.md  (数字人 Prompt)
  - prompts/venue_narrator_v1.md       (讲解员 Prompt)
"""

import os
# =========【必须放在所有transformers导入之前！】=========
# 适配transformers5.x，消除各类警告、路径识别bug
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

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
from llm_client import LLMClient
from query_rewriter import QueryRewriter, RewriteResult

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


def get_embedding(text: str) -> list:
    """查询向量化 — 与 ingest.py 保持 Mean Pooling 一致"""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()

# ============================================================
# 业务配置
# ============================================================
CHROMA_STORE = os.path.join(ROOT_DIR, "chroma_db")
COLLECTION_NAME = "luxun_know_base"
TOP_K = 5
MAX_HISTORY_ROUNDS = 5     # 多轮对话保留最近 N 轮

# Prompt 文件
PROMPT_LUXUN_FILE = os.path.join(ROOT_DIR, "prompts", "luxun_digital_human_v2.md")
PROMPT_NARRATOR_FILE = os.path.join(ROOT_DIR, "prompts", "venue_narrator_v1.md")

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
        语义检索 + 可选的 type 领域过滤。

        Args:
            query:         用户问题文本
            domain_filter: 限定检索的 type 列表，如 ["venue", "bio"]；None 表示全量
            top_k:         返回条数

        Returns:
            [{"content": str, "title": str, "type": str, "source": str,
              "year": int, "distance": float}, ...]
        """
        vec = get_embedding(query)

        # ChromaDB where 过滤
        where_clause = None
        if domain_filter:
            where_clause = {"type": {"$in": domain_filter}}

        res = self.collection.query(
            query_embeddings=[vec],
            n_results=top_k,
            where=where_clause,
        )

        # 结构化返回
        results = []
        if res["ids"] and res["ids"][0]:
            for i in range(len(res["ids"][0])):
                meta = res["metadatas"][0][i] if res["metadatas"] else {}
                results.append({
                    "chunk_id": res["ids"][0][i],
                    "content": res["documents"][0][i],
                    "type": meta.get("type", "unknown"),
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "year": meta.get("year", 0),
                    "character_voice": meta.get("character_voice", False),
                    "venue_relevant": meta.get("venue_relevant", False),
                    "distance": res["distances"][0][i] if res["distances"] else 0.0,
                })
        return results


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
        self.history.append({"role": "user", "content": user})
        self.history.append({"role": "assistant", "content": assistant})
        self.turn_count += 1

        # 只保留最近 MAX_HISTORY_ROUNDS 轮
        max_messages = MAX_HISTORY_ROUNDS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_recent_qa(self) -> Optional[str]:
        if len(self.history) >= 2:
            last_user = self.history[-2]["content"]
            last_assistant = self.history[-1]["content"]
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
        self.state = ConversationState()

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

        # ── Step 1: 意图分类 (Layer 1) ──
        if force_mode is not None:
            intent = force_mode
            intent_result = {"intent": force_mode, "confidence": 1.0, "reason": "manual_force"}
            self.state.current_intent = intent
        else:
            intent_result = classify(user_input)
            intent = intent_result["intent"]
            self.state.current_intent = intent

        if verbose:
            logger.info("意图: %s (conf=%.2f) reason=%s",
                        intent, intent_result["confidence"], intent_result["reason"])
            if intent_result.get("matched"):
                logger.info("  命中: %s", intent_result["matched"][:3])

        # ── Step 2: 按意图路由 ──

        # 2a. 无关/恶意 → 直接拒绝
        if intent == "reject_irrelevant":
            reply = REJECT_IRRELEVANT_RESPONSE
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
            # 用改写后的子查询检索，意图统一按"当前意图或 ambiguous"处理
            search_intent = intent if intent not in ("reject_time",) else "ambiguous"
            chunks = self._retrieve_multi(
                rewrite_result.sub_queries, search_intent
            )

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
        try:
            if len(self.state.history) >= 2:
                # 多轮对话模式
                messages = self.state.to_messages(system_prompt)
                user_content = (
                    f"【参考资料】\n{context_str}\n\n"
                    f"【用户问题】\n{user_input}"
                ) if context_str else user_input
                messages.append({"role": "user", "content": user_content})
                reply = self.llm.call(messages)
            else:
                # 首轮：用 chat 方法
                reply = self.llm.chat(
                    system_prompt=system_prompt,
                    context=context_str,
                    user_query=user_input,
                )
        except Exception as e:
            logger.error("LLM 调用失败: %s", e)
            reply = f"（回答生成失败：{e}）"

        # ── Step 4: 保存对话历史 ──
        self.state.add(user_input, reply)

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
        print("║  鲁迅数字人 · 双模式对话系统                            ║")
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