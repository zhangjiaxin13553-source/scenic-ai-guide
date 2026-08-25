"""
幻觉检测器
==================
从 LLM 回答中抽取事实声明 → 与检索上下文比对 → 标记未支撑声明。

核心流程：
  Step 1: NER 抽取 — 人名/地名/时间点/作品名/事件名/数字
  Step 2: 事实结构化 — (主体, 属性, 值, 原文片段)
  Step 3: 上下文比对 — 精确/模糊/语义匹配
  Step 4: 分类裁决 — VERIFIED | PLAUSIBLE | UNSUPPORTED | CONTRADICTED

输出: HallucinationReport {score, flags, summary}

设计原则：
  - 不引入新模型（用 jieba + 正则 + 复用 BGE embedding）
  - 宽容策略：PLAUSIBLE 不阻塞，仅 UNSUPPORTED + CONTRADICTED 触发警告
  - 阈值可调：HALLUCINATION_THRESHOLD 控制严格度

用法：
  from hallucination_checker import HallucinationChecker
  checker = HallucinationChecker()
  report = checker.check(llm_response, retrieved_context, user_query)
  print(report.score, report.flags, report.summary)
"""

import os
import re
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

import jieba
import jieba.posseg  # noqa: F401 — 显式加载词性标注子模块，否则 jieba.posseg 不可用

logger = logging.getLogger("hallucination_checker")

# ============================================================
# 配置常量
# ============================================================

# 事实声明的裁决阈值（语义相似度）
SIM_THRESHOLD_VERIFIED = 0.82      # ≥ 此值 → VERIFIED
SIM_THRESHOLD_PLAUSIBLE = 0.65     # ≥ 此值 → PLAUSIBLE，低于则 UNSUPPORTED

# 幻觉评分权重
WEIGHT_UNSUPPORTED = 0.25           # 每个 UNSUPPORTED 扣分
WEIGHT_CONTRADICTED = 0.40          # 每个 CONTRADICTED 扣分

# 最小事实声明数：如果回答太短、抽取不到事实，不扣分
MIN_FACTS_FOR_PENALTY = 2

# ============================================================
# 实体/事实抽取规则
# ============================================================

# 书名号模式：《...》
RE_BOOK_TITLE = re.compile(r'《([^》]+)》')

# 引号内容模式
RE_QUOTED = re.compile(r'"([^"]+)"|“([^”]+)”')

# 年份模式
RE_YEAR = re.compile(r'(?:公元)?(\d{4})年(?:代|间|前后|左右)?')

# 具体日期模式
RE_DATE = re.compile(r'(\d{4})年(\d{1,2})月(\d{1,2})日?')

# 数字+量词模式（可能的事实数字）
RE_NUMBER_WITH_UNIT = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:次|封|篇|首|部|个|位|名|所|条|张|本|卷|集|岁|周年)'
)

# 数字范围模式
RE_NUMBER_RANGE = re.compile(r'(\d+)\s*[-~至到]\s*(\d+)')

# 人名模式（基于 jieba 词性标注 + 常见人名后缀）
PERSON_SUFFIX = re.compile(r'(先生|女士|教授|老师|医生|作家|诗人|学者|革命家|思想家)$')

# 地点模式 — 常见地名后缀
PLACE_SUFFIX = re.compile(r'(省|市|县|区|镇|村|路|街|巷|弄|里|馆|楼|堂|园|山|河|湖|海)$')

# AI 腔/模板化句式 — 这些句子不含可核查事实，应跳过
AI_TEMPLATE_PATTERNS = [
    re.compile(r'根据.*(?:资料|记载|记录|研究|说法|文献|史料)'),
    re.compile(r'(?:值得|需要|可以|应该).*(?:注意|关注|思考|了解|认识)'),
    re.compile(r'(?:总体|综合|概括).*来说'),
    re.compile(r'(?:首先|其次|最后|第一|第二|第三)[，,]'),
    re.compile(r'以上.*(?:就是|便是|即是).*(?:介绍|回答|说明)'),
    re.compile(r'(?:希望|但愿|盼).*(?:对您|对你|能).*(?:帮助|有用|启发)'),
    re.compile(r'(?:我虽然|我虽).*(?:但是|但也|却也)'),
]

# 鲁迅常用虚词/句式 — 这些是风格特征，不是事实声明
LUXUN_STYLE_PATTERNS = [
    re.compile(r'(?:大抵|大约|也许|恐怕|或许|或者|似乎|仿佛).*(?:是|有|可以|便是|如此)'),
    re.compile(r'(?:然而|但是|但|却|竟|倒|反倒).*'),
    re.compile(r'(?:我想|我以为|我总觉得|在我的记忆中).*'),
    re.compile(r'(?:这|那).*(?:缘故|道理|意思|说法).*'),
]

# 拒绝语境标记 — 拒绝式回答中为解释拒绝而提及的实体，不属于事实断言
# （与 consistency_checker 的"拒绝语境豁免"保持一致，避免正确拒绝被误判为无支撑幻觉）
_REFUSAL_HINTS = [
    "不知道", "不晓得", "不知晓", "不懂", "不了解", "不清楚", "不熟悉",
    "未曾听说", "未曾见过", "未曾听闻", "不认识", "没听说",
    "不能回答", "无法回答", "无从回答", "难以回答", "无从",
    "不是我所能", "非我所知", "属于另一个时代", "身后", "生前", "死后",
    "你问的", "你说的", "你所问", "提到的", "所谓", "问的是",
]


def _in_refusal_context(text: str, idx: int, window: int = 40) -> bool:
    """判断某位置 idx 前后 window 字符内是否处于拒绝/转述语境。"""
    seg = text[max(0, idx - window): min(len(text), idx + window)]
    return any(m in seg for m in _REFUSAL_HINTS)

# ============================================================
# 数据结构
# ============================================================

@dataclass
class FactClaim:
    """一条事实声明"""
    subject: str           # 主体（鲁迅/《狂人日记》/广州鲁迅纪念馆）
    attribute: str         # 属性（发表时间/作者/地点）
    value: str             # 值（1918年5月/鲁迅/广州）
    snippet: str           # 原文片段（便于人工复核）
    category: str          # 类别: person / work / time / place / number / event
    verdict: str = ""      # VERIFIED | PLAUSIBLE | UNSUPPORTED | CONTRADICTED
    evidence: str = ""     # 匹配到的上下文片段
    sim_score: float = 0.0 # 语义相似度

@dataclass
class HallucinationReport:
    """幻觉检测报告"""
    score: float                        # 0.0~1.0 可靠性评分
    flags: List[FactClaim] = field(default_factory=list)    # 标记的声明
    all_claims: List[FactClaim] = field(default_factory=list)
    summary: str = ""                   # 一句话总结
    details: dict = field(default_factory=dict)


# ============================================================
# HallucinationChecker
# ============================================================

class HallucinationChecker:
    """
    幻觉检测器。

    从 LLM 回答中抽取事实声明，与检索上下文逐一比对，
    输出可靠性评分 + 标记列表。

    使用方式：
        checker = HallucinationChecker()
        report = checker.check(
            llm_response="鲁迅生于1881年...",
            retrieved_context=[{"content": "...", ...}, ...],
            user_query="鲁迅是哪一年出生的？"
        )
    """

    def __init__(
        self,
        sim_verified: float = SIM_THRESHOLD_VERIFIED,
        sim_plausible: float = SIM_THRESHOLD_PLAUSIBLE,
        weight_unsupported: float = WEIGHT_UNSUPPORTED,
        weight_contradicted: float = WEIGHT_CONTRADICTED,
    ):
        self.sim_verified = sim_verified
        self.sim_plausible = sim_plausible
        self.weight_unsupported = weight_unsupported
        self.weight_contradicted = weight_contradicted

        # BGE embedding 模型（延迟加载，与项目统一）
        self._embedder = None

    # ---- embedding 延迟加载 ----

    @property
    def embedder(self):
        """延迟加载 BGE 模型，复用项目统一的 embedding"""
        if self._embedder is None:
            from transformers import AutoTokenizer, AutoModel
            import torch

            SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
            LOCAL_MODEL_PATH = os.path.join(SCRIPT_DIR, "bge-small-zh-v1.5")
            HF_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

            if os.path.isdir(LOCAL_MODEL_PATH):
                model_path = LOCAL_MODEL_PATH
            else:
                model_path = HF_MODEL_NAME

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModel.from_pretrained(model_path).to(self._device)
            self._model.eval()
            self._embedder = True  # 标记已初始化

        return self._embedder

    def _encode(self, text: str) -> list:
        """用 BGE 模型编码文本为向量"""
        import torch
        self.embedder  # 确保已初始化
        inputs = self._tokenizer(
            text, padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()

    def _cosine_sim(self, a: list, b: list) -> float:
        """计算两个向量的余弦相似度"""
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # ---- Step 1: 事实抽取 ----

    def _extract_entities(self, text: str) -> List[FactClaim]:
        """从回答文本中抽取可核查的事实声明"""
        claims = []

        # ----- 1.1 作品名（书名号）-----
        book_titles = RE_BOOK_TITLE.findall(text)
        for title in book_titles:
            # 找书名附近的上下文
            idx = text.find(f'《{title}》')
            context_start = max(0, idx - 30)
            context_end = min(len(text), idx + len(title) + 35)
            snippet = text[context_start:context_end].strip()

            claims.append(FactClaim(
                subject=title,
                attribute="作品名",
                value=title,
                snippet=snippet,
                category="work",
            ))

        # ----- 1.2 年份/日期 -----
        years = RE_YEAR.findall(text)
        for yr in years:
            yr_int = int(yr)
            # 跳过太远古的年份（如"公元前"）和太近的年份
            if yr_int < 1800 or yr_int > 1950:
                continue
            idx = text.find(f'{yr}年')
            context_start = max(0, idx - 30)
            context_end = min(len(text), idx + 35)
            snippet = text[context_start:context_end].strip()

            # 判断这个年份关联的主体
            subject = self._find_subject_near(text, idx, window=40)

            claims.append(FactClaim(
                subject=subject or "鲁迅",
                attribute="时间",
                value=f"{yr}年",
                snippet=snippet,
                category="time",
            ))

        # 具体日期
        dates = RE_DATE.findall(text)
        for yr, mo, day in dates:
            full_date = f"{yr}年{mo}月{day}日"
            idx = text.find(full_date)
            context_start = max(0, idx - 30)
            context_end = min(len(text), idx + 40)
            snippet = text[context_start:context_end].strip()
            subject = self._find_subject_near(text, idx, window=50)

            claims.append(FactClaim(
                subject=subject or "鲁迅",
                attribute="具体日期",
                value=full_date,
                snippet=snippet,
                category="time",
            ))

        # ----- 1.3 数字+量词 -----
        numbers = RE_NUMBER_WITH_UNIT.findall(text)
        for num in numbers:
            idx = text.find(num)
            context_start = max(0, idx - 25)
            context_end = min(len(text), idx + 30)
            snippet = text[context_start:context_end].strip()
            subject = self._find_subject_near(text, idx, window=40)

            claims.append(FactClaim(
                subject=subject or "未知",
                attribute="数量",
                value=num,
                snippet=snippet,
                category="number",
            ))

        # ----- 1.4 人名（基于 jieba 分词）-----
        words = jieba.posseg.cut(text)
        for w, flag in words:
            if flag == "nr" and len(w) >= 2:  # 人名
                # 过滤掉"鲁迅"本身（回答中必然大量出现）
                if w in ("鲁迅", "周树人", "我"):
                    continue
                idx = text.find(w)
                # 拒绝语境豁免：拒绝/坦承不知时提及的人名，不是事实断言
                if _in_refusal_context(text, idx):
                    continue
                context_start = max(0, idx - 20)
                context_end = min(len(text), idx + 25)
                snippet = text[context_start:context_end].strip()

                claims.append(FactClaim(
                    subject=w,
                    attribute="人物",
                    value=w,
                    snippet=snippet,
                    category="person",
                ))

        # ----- 1.5 地点 -----
        words2 = jieba.posseg.cut(text)
        for w, flag in words2:
            if flag == "ns" and len(w) >= 2:  # 地名
                # 过滤掉"广州"（场馆相关，必然出现）
                if w in ("广州", "中国"):
                    continue
                idx = text.find(w)
                # 拒绝语境豁免：拒绝时提及的地名，不是事实断言
                if _in_refusal_context(text, idx):
                    continue
                context_start = max(0, idx - 20)
                context_end = min(len(text), idx + 25)
                snippet = text[context_start:context_end].strip()

                claims.append(FactClaim(
                    subject=w,
                    attribute="地点",
                    value=w,
                    snippet=snippet,
                    category="place",
                ))

        return claims

    def _find_subject_near(self, text: str, pos: int, window: int = 40) -> Optional[str]:
        """在文本中查找 pos 位置附近的名词作为主体"""
        chunk = text[max(0, pos - window): min(len(text), pos + window)]
        # 找最近的书名
        books = RE_BOOK_TITLE.findall(chunk)
        if books:
            return books[-1]
        # 找最近的人名
        words = jieba.posseg.cut(chunk)
        persons = [w for w, flag in words if flag == "nr" and len(w) >= 2]
        if persons:
            return persons[-1]
        return None

    # ---- Step 2: 上下文比对 ----

    def _verify_claim(
        self, claim: FactClaim, context_texts: List[str]
    ) -> Tuple[str, str, float]:
        """
        将一条事实声明与检索上下文比对。

        Returns:
            (verdict, evidence, sim_score)
        """
        if not context_texts:
            return ("UNSUPPORTED", "", 0.0)

        claim_text = claim.snippet or f"{claim.subject}{claim.value}"

        # 先精确匹配
        for ctx in context_texts:
            if claim.value in ctx and len(claim.value) >= 2:
                return ("VERIFIED", ctx[:200], 1.0)

        # 模糊匹配 — 语义相似度
        best_sim = 0.0
        best_ctx = ""
        try:
            claim_vec = self._encode(claim_text)
            for ctx in context_texts:
                ctx_vec = self._encode(ctx[:512])
                sim = self._cosine_sim(claim_vec, ctx_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_ctx = ctx[:200]
        except Exception as e:
            logger.warning("语义比对失败: %s，退回精确匹配", e)
            return ("UNSUPPORTED", "", 0.0)

        if best_sim >= self.sim_verified:
            return ("VERIFIED", best_ctx, best_sim)
        elif best_sim >= self.sim_plausible:
            return ("PLAUSIBLE", best_ctx, best_sim)
        else:
            return ("UNSUPPORTED", best_ctx[:200] if best_ctx else "", best_sim)

    # ---- Step 3: 过滤非事实性声明 ----

    def _is_factual_claim(self, claim: FactClaim) -> bool:
        """判断一条声明是否为可核查的事实声明（排除风格/套话）"""
        snippet = claim.snippet

        # 跳过 AI 模板句式
        for pat in AI_TEMPLATE_PATTERNS:
            if pat.search(snippet):
                return False

        # 跳过鲁迅风格句式
        for pat in LUXUN_STYLE_PATTERNS:
            if pat.search(snippet):
                return False

        # 作品名如果只是"提到"而非"断言事实"，跳过
        if claim.category == "work":
            # 如果 snippet 只包含书名号而无其他断言，跳过
            clean = snippet.replace(claim.value, "").replace(f'《{claim.value}》', "")
            if len(clean.strip()) < 10:
                return False

        return True

    # ---- 主入口 ----

    def check(
        self,
        llm_response: str,
        retrieved_context: List[dict],
        user_query: str = "",
    ) -> HallucinationReport:
        """
        对 LLM 回答执行幻觉检测。

        Args:
            llm_response:       LLM 生成的回答文本
            retrieved_context:  检索到的知识片段列表
                                [{"content": "...", ...}, ...]
            user_query:         用户原始问题（可选，用于上下文）

        Returns:
            HallucinationReport {score, flags, all_claims, summary, details}
        """
        if not llm_response or not llm_response.strip():
            return HallucinationReport(
                score=1.0,
                summary="回答为空，无幻觉风险",
            )

        # 提取上下文文本
        context_texts = [
            c.get("content", "") or c.get("snippet", "") or ""
            for c in (retrieved_context or [])
        ]
        context_texts = [t for t in context_texts if t]  # 过滤空串

        # Step 1: 抽取事实声明
        raw_claims = self._extract_entities(llm_response)

        # Step 2: 过滤非事实性声明
        factual_claims = [c for c in raw_claims if self._is_factual_claim(c)]

        # Step 3: 逐条比对
        for claim in factual_claims:
            verdict, evidence, sim = self._verify_claim(claim, context_texts)
            claim.verdict = verdict
            claim.evidence = evidence
            claim.sim_score = sim

        # Step 4: 统计裁决
        verified = [c for c in factual_claims if c.verdict == "VERIFIED"]
        plausible = [c for c in factual_claims if c.verdict == "PLAUSIBLE"]
        unsupported = [c for c in factual_claims if c.verdict == "UNSUPPORTED"]
        contradicted = [c for c in factual_claims if c.verdict == "CONTRADICTED"]

        # 计算评分
        total_factual = len(factual_claims)
        if total_factual < MIN_FACTS_FOR_PENALTY:
            # 回答太短或没有可核查的事实声明，不扣分
            score = 1.0
        else:
            score = 1.0
            score -= len(unsupported) * self.weight_unsupported
            score -= len(contradicted) * self.weight_contradicted
            score = max(0.0, min(1.0, score))

        # 汇总标记（仅 UNSUPPORTED + CONTRADICTED）
        flags = unsupported + contradicted

        # 生成摘要
        parts = []
        if verified:
            parts.append(f"已验证 {len(verified)} 条事实")
        if unsupported:
            parts.append(f"⚠ {len(unsupported)} 条无支撑")
        if contradicted:
            parts.append(f"❌ {len(contradicted)} 条矛盾")
        if not parts:
            parts.append("未抽取到可核查事实声明")

        summary = "；".join(parts)

        return HallucinationReport(
            score=score,
            flags=flags,
            all_claims=factual_claims,
            summary=summary,
            details={
                "total_raw": len(raw_claims),
                "total_factual": total_factual,
                "verified": len(verified),
                "plausible": len(plausible),
                "unsupported": len(unsupported),
                "contradicted": len(contradicted),
                "context_chunks": len(context_texts),
            },
        )

    # ---- 便捷方法：直接对 RAG 返回结果做检查 ----

    def check_rag_result(
        self,
        reply: str,
        chunks: List[dict],
        query: str = "",
    ) -> HallucinationReport:
        """对 RAGPipeline.ask() 的返回结果直接做幻觉检查"""
        return self.check(
            llm_response=reply,
            retrieved_context=chunks,
            user_query=query,
        )


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-5s | %(message)s",
    )

    print("=" * 60)
    print("  幻觉检测器 自测")
    print("=" * 60)

    checker = HallucinationChecker()

    # 模拟场景
    context = [
        {
            "content": (
                "鲁迅（1881年9月25日－1936年10月19日），原名周树人，"
                "字豫才，浙江绍兴人。中国现代文学的奠基人之一。"
                "1918年5月，他在《新青年》第四卷第五号上发表了《狂人日记》，"
                "这是中国第一篇白话短篇小说。"
            ),
            "title": "鲁迅生平",
            "type": "bio",
        },
        {
            "content": (
                "《呐喊》是鲁迅的第一部短篇小说集，1923年8月由北京新潮社出版。"
                "收录了1918年至1922年间创作的14篇小说，"
                "包括《狂人日记》《孔乙己》《药》《故乡》《阿Q正传》等。"
            ),
            "title": "《呐喊》",
            "type": "work",
        },
    ]

    test_cases = [
        # Case 1: 有据可查的回答（应高分）
        (
            "鲁迅原名周树人，1881年9月25日出生于浙江绍兴。"
            "1918年5月，他在《新青年》上发表了《狂人日记》。",
            "准确回答，应高分",
        ),
        # Case 2: 编造了具体日期
        (
            "《狂人日记》发表在1918年5月15日的《新青年》第四卷第五号上，"
            "当时鲁迅37岁。这篇作品标志着中国现代文学的开端。",
            "编造了5月15日具体日期，应标记",
        ),
        # Case 3: 编造了知识库没有的数字
        (
            "鲁迅一生总共写了约650篇文章，其中包括23部小说集、"
            "17部杂文集。他的作品在全世界被翻译成了58种语言。",
            "编造了文章数量，应大量标记",
        ),
        # Case 4: 安全回答（模糊处理）
        (
            "关于这个问题，我手头的资料有限。鲁迅的作品确有不少，"
            "但具体的数字恐怕难以精确计算。",
            "诚实承认未知，应高分",
        ),
    ]

    for i, (response, desc) in enumerate(test_cases, 1):
        print(f"\n--- Case {i}: {desc} ---")
        report = checker.check(response, context)

        print(f"  评分: {report.score:.2f}")
        print(f"  摘要: {report.summary}")
        if report.flags:
            print(f"  标记 ({len(report.flags)} 条):")
            for f in report.flags:
                print(f"    [{f.verdict}] {f.category}: {f.value}")
                print(f"      原文: {f.snippet[:80]}...")
        print(f"  详情: {report.details}")
