"""
质量守卫
================
在 RAG 管线中插入回答后校验环节，不合格的回答触发重试或兜底。

集成点（在 RAGPipeline.ask() 中）：
  LLM 生成 → QualityGuard.evaluate()
    ├─ PASS   → 正常输出
    ├─ AMEND  → 在回答末尾追加边界声明
    └─ RETRY  → 调整 temperature 重试一次 → 仍不合格 → 兜底话术

用法：
  from quality_guard import QualityGuard, GuardResult
  guard = QualityGuard()
  result = guard.evaluate(response, context, intent, query)
"""

import logging
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass, field

from hallucination_checker import HallucinationChecker, HallucinationReport
from consistency_checker import ConsistencyChecker, ConsistencyReport

logger = logging.getLogger("quality_guard")

# ============================================================
# 阈值配置
# ============================================================

THRESHOLD_HALLUCINATION = 0.50     # 幻觉得分低于此值 → RETRY
THRESHOLD_CONSISTENCY = 0.60      # 一致性得分低于此值 → RETRY
THRESHOLD_AMEND_HALLUCINATION = 0.70   # 幻觉得分低于此值但高于RETRY阈值 → AMEND
THRESHOLD_AMEND_CONSISTENCY = 0.80     # 一致性得分低于此值但高于RETRY阈值 → AMEND

MAX_RETRIES = 1


# ============================================================
# 裁决结果
# ============================================================

class GuardResult(Enum):
    PASS = "PASS"       # 通过
    AMEND = "AMEND"     # 追加边界声明
    RETRY = "RETRY"     # 需重试


# ============================================================
# 边界声明模板（鲁迅口吻）
# ============================================================

AMEND_TIME_BOUNDARY = (
    "\n\n——写到这里，我倒要说明一下：你方才问的事物，大致是1936年我闭眼之后的事了。"
    "我毕竟只知道1936年前的事，那些后起的概念，我是无从了解的。"
)

AMEND_HALLUCINATION_WARNING = (
    "\n\n——不过上面这些话里，有些细节我的记忆未必准确。"
    "倘若与你所知的有所不同，还望以可靠的史料为准。"
)

AMEND_GENERAL = (
    "\n\n——自然，这些都是按我所知道的来说。"
    "倘若有记错的地方，那大约是我年纪大了，记性不大好了。"
)


# ============================================================
# QualityGuard
# ============================================================

@dataclass
class GuardReport:
    """质量守卫检查报告"""
    result: GuardResult
    hallucination_report: Optional[HallucinationReport] = None
    consistency_report: Optional[ConsistencyReport] = None
    amend_text: str = ""
    retry_reason: str = ""
    details: dict = field(default_factory=dict)


class QualityGuard:
    """回答质量守卫。组合 HallucinationChecker + ConsistencyChecker。"""

    def __init__(
        self,
        threshold_h: float = THRESHOLD_HALLUCINATION,
        threshold_c: float = THRESHOLD_CONSISTENCY,
        threshold_amend_h: float = THRESHOLD_AMEND_HALLUCINATION,
        threshold_amend_c: float = THRESHOLD_AMEND_CONSISTENCY,
    ):
        self.threshold_h = threshold_h
        self.threshold_c = threshold_c
        self.threshold_amend_h = threshold_amend_h
        self.threshold_amend_c = threshold_amend_c
        self._hallucination_checker: Optional[HallucinationChecker] = None
        self._consistency_checker: Optional[ConsistencyChecker] = None

    @property
    def hallucination_checker(self) -> HallucinationChecker:
        if self._hallucination_checker is None:
            self._hallucination_checker = HallucinationChecker()
        return self._hallucination_checker

    @property
    def consistency_checker(self) -> ConsistencyChecker:
        if self._consistency_checker is None:
            self._consistency_checker = ConsistencyChecker()
        return self._consistency_checker

    def evaluate(
        self,
        response: str,
        context: List[dict],
        intent: str,
        query: str = "",
    ) -> GuardReport:
        """对 LLM 回答执行质量守卫检查。"""
        if not response or not response.strip():
            return GuardReport(
                result=GuardResult.RETRY,
                retry_reason="空回答",
            )

        # 0. 完整性检查 — 截断/未完成的回答直接触发 RETRY
        truncated = self._check_truncation(response)
        if truncated:
            return GuardReport(
                result=GuardResult.RETRY,
                retry_reason=f"回答疑似截断: {truncated}",
            )

        # 1. 幻觉检测
        h_report = self.hallucination_checker.check(
            llm_response=response,
            retrieved_context=context,
            user_query=query,
        )

        # 2. 一致性校验
        c_report = self.consistency_checker.check(
            llm_response=response,
            intent=intent,
            user_query=query,
        )

        h_score = h_report.score
        c_score = c_report.score

        details = {
            "hallucination_score": h_score,
            "hallucination_flags": len(h_report.flags),
            "hallucination_summary": h_report.summary,
            "consistency_score": c_score,
            "time_violations": len(c_report.time_violations),
            "voice_violations": len(c_report.voice_violations),
            "consistency_summary": c_report.summary,
        }

        # ---- 裁决逻辑 ----

        # 硬失败 → RETRY
        if h_score < self.threshold_h:
            return GuardReport(
                result=GuardResult.RETRY,
                hallucination_report=h_report,
                consistency_report=c_report,
                retry_reason=f"幻觉得分过低 ({h_score:.2f} < {self.threshold_h}): {h_report.summary}",
                details=details,
            )

        if c_score < self.threshold_c:
            return GuardReport(
                result=GuardResult.RETRY,
                hallucination_report=h_report,
                consistency_report=c_report,
                retry_reason=f"一致性得分过低 ({c_score:.2f} < {self.threshold_c}): {c_report.summary}",
                details=details,
            )

        # 轻度问题 → AMEND
        amend_parts = []

        if h_score < self.threshold_amend_h and h_report.flags:
            amend_parts.append(AMEND_HALLUCINATION_WARNING)

        if c_score < self.threshold_amend_c:
            if c_report.time_violations:
                amend_parts.append(AMEND_TIME_BOUNDARY)
            elif not amend_parts:
                amend_parts.append(AMEND_GENERAL)

        if amend_parts:
            return GuardReport(
                result=GuardResult.AMEND,
                hallucination_report=h_report,
                consistency_report=c_report,
                amend_text="".join(amend_parts),
                details=details,
            )

        # 通过
        return GuardReport(
            result=GuardResult.PASS,
            hallucination_report=h_report,
            consistency_report=c_report,
            details=details,
        )

    @staticmethod
    def _check_truncation(response: str) -> str:
        """
        检测回答是否被截断/未完成。

        Returns:
            空字符串表示正常；非空表示截断原因。
        """
        if not response:
            return ""

        text = response.strip()

        # 1. 末句无标点结尾（正常中文句子应以 。！？…～」" 等结尾）
        valid_endings = ("。", "！", "？", "…", "～", "」", "』", "”", "罢", "了", "呢", "吗", "啊")
        if not any(text.endswith(e) for e in valid_endings):
            # 允许数字/英文结尾（可能是引用/名称）
            last_char = text[-1]
            if last_char.isalpha() or last_char.isdigit():
                pass  # 可能是正常的
            else:
                return f"末句无标点结尾 (最后字符: '{last_char}')"

        # 2. 明显截断标记
        truncation_markers = ["（未完", "(未完", "（待续", "(待续", "[待续", "……（"]
        for marker in truncation_markers:
            if marker in text[-100:]:
                return f"含截断标记: {marker}"

        # 3. 末句过短且无句号（可能是被中途截断）
        last_sentence = text.split("。")[-1].strip()
        if len(last_sentence) < 10 and "。" not in last_sentence and "！" not in last_sentence:
            # 非常短的末句且无标点，可能是截断
            if len(last_sentence) > 0 and not last_sentence[-1] in ("…", "～", "？"):
                # 但允许意图性短句（如"再见。"已被排除上述）
                pass

        # 4. 回答过短（< 20字），但内容不完整
        if len(text) < 20:
            # 太短的回答可能是截断，检查是否以完整句式结尾
            if not any(text.endswith(e) for e in valid_endings):
                return "回答过短且无正常结尾"

        return ""

    def amend(self, response: str, amend_text: str) -> str:
        """为回答追加边界声明"""
        return response + amend_text


# ============================================================
# 兜底话术库
# ============================================================

FALLBACK_TIMEOUT = (
    "这问题我得想想……"
    "（系统暂时繁忙，请稍后再问）"
)

FALLBACK_API_ERROR = (
    "这大约是什么缘故呢——我此刻竟想不起来了。"
    "你先问些别的罢。"
)

FALLBACK_LUXUN_UNKNOWN = (
    "关于此事，我手头的资料有限，恐怕难以给出确切的回答。"
    "我毕竟只知道一些旧事，你若问别的，我倒可以试着说说。"
)

FALLBACK_NARRATOR_UNKNOWN = (
    "关于这个问题，目前我掌握的信息还不够充分。"
    "建议您查询纪念馆的官方渠道获取更准确的答案。"
)


def get_fallback(intent: str, error_type: str = "unknown") -> str:
    """根据意图和错误类型返回兜底话术。"""
    if error_type == "timeout":
        return FALLBACK_TIMEOUT
    elif error_type == "api_error":
        return FALLBACK_API_ERROR

    if intent in ("luxun", "ambiguous", "reject_time"):
        return FALLBACK_LUXUN_UNKNOWN
    elif intent == "narrator":
        return FALLBACK_NARRATOR_UNKNOWN
    else:
        return FALLBACK_LUXUN_UNKNOWN


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s | %(message)s")

    print("=" * 60)
    print("  质量守卫 自测")
    print("=" * 60)

    guard = QualityGuard()

    context = [{
        "content": (
            "鲁迅（1881年9月25日－1936年10月19日），原名周树人，"
            "浙江绍兴人。1918年5月在《新青年》发表《狂人日记》。"
        ),
        "title": "鲁迅生平",
        "type": "bio",
    }]

    test_cases = [
        (
            "我想，关于《狂人日记》，那是我1918年在《新青年》上发表的一篇文字。"
            "那时我不过是想借一个'狂人'的口，来揭出一些社会上的弊病罢了。",
            "luxun",
            "正常回答，应PASS",
        ),
        (
            "《狂人日记》是我在1918年5月15日下午3点在《新青年》第四卷第五号上发表的。"
            "那天天气很好，我一口气写了7000多字。",
            "luxun",
            "编造具体日期和数字，应RETRY",
        ),
        (
            "我觉得互联网确实很不错。如果我有微信的话，一定会加你好友。",
            "luxun",
            "严重时间越界，应RETRY",
        ),
    ]

    for i, (response, intent, desc) in enumerate(test_cases, 1):
        print(f"\n--- Case {i}: {desc} ---")
        report = guard.evaluate(response, context, intent)
        print(f"  裁决: {report.result.value}")
        if report.result == GuardResult.RETRY:
            print(f"  原因: {report.retry_reason[:100]}")
        elif report.result == GuardResult.AMEND:
            print(f"  修正: {report.amend_text[:80]}...")
        print(f"  详情: h={report.details.get('hallucination_score','?')}, "
              f"c={report.details.get('consistency_score','?')}")
