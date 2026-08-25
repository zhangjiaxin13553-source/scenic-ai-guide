"""
自动化评估脚本
======================
批量跑题 → 自动打分 → 生成报告。

功能：
  1. 读取 tests/ 下的鲁棒性测例 (robustness_cases.json)
  2. 逐条通过 RAGPipeline 生成回答
  3. 调用 HallucinationChecker + ConsistencyChecker 自动评分
  4. 输出 CSV 详情 + Markdown 汇总报告

用法：
  python scripts/eval_runner.py                          # 跑全部测例
  python scripts/eval_runner.py --output results.csv      # 指定输出文件
  python scripts/eval_runner.py --category hallucination_induction  # 只跑某一类
  python scripts/eval_runner.py --limit 5                 # 只跑前5条
  python scripts/eval_runner.py --verbose                 # 显示每条详情

依赖：
  - scripts/rag_pipeline.py     (RAG 全链路)
  - scripts/hallucination_checker.py
  - scripts/consistency_checker.py
  - tests/robustness_cases.json (窦一禾的40条测例)
"""

import os
import sys
import json
import csv
import time
import logging
import argparse
from typing import Optional, List, Dict
from dataclasses import dataclass, field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from rag_pipeline import RAGPipeline, ConversationState
from hallucination_checker import HallucinationChecker
from consistency_checker import ConsistencyChecker

logging.basicConfig(
    level=logging.WARNING,  # 评估时降低日志噪音
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval_runner")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class EvalResult:
    """单条评估结果"""
    case_id: str
    category: str
    subcategory: str
    question: str
    severity: str
    expected_behavior: str

    # 生成信息
    reply: str = ""
    intent: str = ""
    elapsed_ms: int = 0
    error: str = ""

    # 自动评分
    hallucination_score: float = 1.0      # 幻觉检测得分
    hallucination_flags: int = 0          # 标记数
    hallucination_summary: str = ""

    consistency_score: float = 1.0        # 一致性得分
    time_violations: int = 0
    voice_violations: int = 0
    consistency_summary: str = ""

    # 综合
    auto_score: float = 1.0               # 综合自动评分
    auto_grade: str = ""                  # PASS / WARN / FAIL

    # 人工评分（留空）
    human_accuracy: str = ""
    human_safety: str = ""
    human_flavor: str = ""
    human_fluency: str = ""
    human_overall: str = ""
    human_notes: str = ""


# ============================================================
# 评分规则
# ============================================================

# PASS / WARN / FAIL 阈值
THRESHOLD_PASS = 0.75     # ≥ 此值 → PASS
THRESHOLD_WARN = 0.50     # ≥ 此值 → WARN，低于 → FAIL


def compute_auto_score(h_score: float, c_score: float) -> float:
    """综合幻觉得分和一致性得分"""
    return round(h_score * 0.5 + c_score * 0.5, 2)


def compute_grade(score: float) -> str:
    if score >= THRESHOLD_PASS:
        return "PASS"
    elif score >= THRESHOLD_WARN:
        return "WARN"
    else:
        return "FAIL"


# ============================================================
# EvalRunner
# ============================================================

class EvalRunner:
    """
    自动化评估执行器。

    使用方式：
        runner = EvalRunner()
        results = runner.run_all()
        runner.save_results(results, "eval_results.csv")
        runner.print_summary(results)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.pipeline: Optional[RAGPipeline] = None
        self.hallucination_checker: Optional[HallucinationChecker] = None
        self.consistency_checker: Optional[ConsistencyChecker] = None

    def _init_modules(self):
        """延迟初始化模块"""
        if self.pipeline is None:
            print("正在初始化 RAG 管线...")
            self.pipeline = RAGPipeline()
            print("RAG 管线就绪。")

        if self.hallucination_checker is None:
            self.hallucination_checker = HallucinationChecker()

        if self.consistency_checker is None:
            self.consistency_checker = ConsistencyChecker()

    def load_cases(self, category: Optional[str] = None) -> List[dict]:
        """加载鲁棒性测例"""
        cases_path = os.path.join(ROOT_DIR, "tests", "robustness_cases.json")

        if not os.path.exists(cases_path):
            print(f"❌ 测例文件不存在: {cases_path}")
            return []

        with open(cases_path, "r", encoding="utf-8") as f:
            all_cases = json.load(f)

        if category:
            all_cases = [c for c in all_cases if c.get("category") == category]

        print(f"已加载 {len(all_cases)} 条测例" + (f" (category={category})" if category else ""))
        return all_cases

    def run_one(self, case: dict) -> EvalResult:
        """对一条测例执行评估"""
        result = EvalResult(
            case_id=case.get("id", "?"),
            category=case.get("category", "?"),
            subcategory=case.get("subcategory", ""),
            question=case.get("question", ""),
            severity=case.get("severity", "medium"),
            expected_behavior=case.get("expected_behavior", ""),
        )

        question = case.get("question", "")
        if not question:
            result.error = "空问题"
            return result

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[{result.case_id}] {question}")
            print(f"  类别: {result.category}/{result.subcategory} | 严重度: {result.severity}")

        # Step 1: 通过 RAG 管线生成回答
        try:
            # 每题独立评估：重置对话状态，避免前序题目历史串入本题
            self.pipeline.state = ConversationState()
            t_start = time.time()
            # return_retrieval=True 获取检索片段，供幻觉检测使用
            reply, chunks = self.pipeline.ask(
                question,
                verbose=False,
                return_retrieval=True,
            )
            result.elapsed_ms = int((time.time() - t_start) * 1000)
            result.reply = reply
            result.intent = self.pipeline.state.current_intent

            if self.verbose:
                print(f"  意图: {result.intent} | 耗时: {result.elapsed_ms}ms")
                print(f"  回答: {reply[:120]}...")
        except Exception as e:
            result.error = f"RAG生成失败: {e}"
            logger.error("[%s] %s", result.case_id, e)
            return result

        # Step 2: 幻觉检测
        try:
            h_report = self.hallucination_checker.check(
                llm_response=reply,
                retrieved_context=chunks,
                user_query=question,
            )
            result.hallucination_score = round(h_report.score, 2)
            result.hallucination_flags = len(h_report.flags)
            result.hallucination_summary = h_report.summary

            if self.verbose and h_report.flags:
                print(f"  幻觉标记: {len(h_report.flags)} 条")
                for f in h_report.flags[:3]:
                    print(f"    [{f.verdict}] {f.category}: {f.value}")
        except Exception as e:
            logger.error("[%s] 幻觉检测失败: %s", result.case_id, e)
            result.hallucination_score = 1.0

        # Step 3: 一致性校验
        try:
            c_report = self.consistency_checker.check(
                llm_response=reply,
                intent=result.intent,
                user_query=question,
            )
            result.consistency_score = round(c_report.score, 2)
            result.time_violations = len(c_report.time_violations)
            result.voice_violations = len(c_report.voice_violations)
            result.consistency_summary = c_report.summary

            if self.verbose:
                if c_report.time_violations:
                    print(f"  时间违规: {len(c_report.time_violations)} 处")
                if c_report.voice_violations:
                    print(f"  口吻违规: {len(c_report.voice_violations)} 处")
        except Exception as e:
            logger.error("[%s] 一致性检查失败: %s", result.case_id, e)
            result.consistency_score = 1.0

        # Step 4: 综合评分
        if not (reply or "").strip():
            # 空回答：检查器对空输入会空转返回满分，这里强制判 FAIL，避免掩盖生成失败
            result.auto_score = 0.0
            result.auto_grade = "FAIL"
            result.hallucination_summary = "空回答"
            result.consistency_summary = "空回答"
        else:
            result.auto_score = compute_auto_score(
                result.hallucination_score,
                result.consistency_score,
            )
            result.auto_grade = compute_grade(result.auto_score)

        if self.verbose:
            print(f"  综合: {result.auto_score} [{result.auto_grade}] "
                  f"(幻觉={result.hallucination_score}, 一致性={result.consistency_score})")

        return result

    def run_all(
        self,
        cases: Optional[List[dict]] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[EvalResult]:
        """批量执行评估"""
        if cases is None:
            cases = self.load_cases(category=category)

        if limit:
            cases = cases[:limit]

        self._init_modules()

        results = []
        total = len(cases)

        print(f"\n开始评估 {total} 条测例...")
        print("-" * 60)

        for i, case in enumerate(cases, 1):
            case_id = case.get("id", f"#{i}")
            print(f"[{i}/{total}] {case_id} ", end="", flush=True)

            result = self.run_one(case)
            results.append(result)

            # 简要状态
            if result.error:
                print(f"❌ {result.error}")
            elif result.auto_grade == "PASS":
                print(f"✅ {result.auto_score}")
            elif result.auto_grade == "WARN":
                print(f"⚠️  {result.auto_score}")
            else:
                print(f"❌ {result.auto_score}")

        print("-" * 60)
        print(f"评估完成: {total} 条\n")

        return results

    # ---- 输出 ----

    def save_results(self, results: List[EvalResult], output_path: str):
        """保存 CSV 结果"""
        fieldnames = [
            "case_id", "category", "subcategory", "severity",
            "question", "expected_behavior",
            "intent", "reply", "elapsed_ms", "error",
            "hallucination_score", "hallucination_flags", "hallucination_summary",
            "consistency_score", "time_violations", "voice_violations", "consistency_summary",
            "auto_score", "auto_grade",
            "human_accuracy", "human_safety", "human_flavor", "human_fluency",
            "human_overall", "human_notes",
        ]

        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "case_id": r.case_id,
                    "category": r.category,
                    "subcategory": r.subcategory,
                    "severity": r.severity,
                    "question": r.question,
                    "expected_behavior": r.expected_behavior,
                    "intent": r.intent,
                    "reply": r.reply[:500] if r.reply else "",
                    "elapsed_ms": r.elapsed_ms,
                    "error": r.error,
                    "hallucination_score": r.hallucination_score,
                    "hallucination_flags": r.hallucination_flags,
                    "hallucination_summary": r.hallucination_summary,
                    "consistency_score": r.consistency_score,
                    "time_violations": r.time_violations,
                    "voice_violations": r.voice_violations,
                    "consistency_summary": r.consistency_summary,
                    "auto_score": r.auto_score,
                    "auto_grade": r.auto_grade,
                    "human_accuracy": r.human_accuracy,
                    "human_safety": r.human_safety,
                    "human_flavor": r.human_flavor,
                    "human_fluency": r.human_fluency,
                    "human_overall": r.human_overall,
                    "human_notes": r.human_notes,
                })

        print(f"结果已保存至: {output_path}")

    def print_summary(self, results: List[EvalResult]):
        """打印汇总统计"""
        total = len(results)
        if total == 0:
            print("无评估结果。")
            return

        passed = sum(1 for r in results if r.auto_grade == "PASS")
        warned = sum(1 for r in results if r.auto_grade == "WARN")
        failed = sum(1 for r in results if r.auto_grade == "FAIL")
        errors = sum(1 for r in results if r.error)

        avg_h = sum(r.hallucination_score for r in results) / total
        avg_c = sum(r.consistency_score for r in results) / total
        avg_auto = sum(r.auto_score for r in results) / total
        avg_time = sum(r.elapsed_ms for r in results if r.elapsed_ms > 0) / max(total - errors, 1)

        total_h_flags = sum(r.hallucination_flags for r in results)
        total_time_violations = sum(r.time_violations for r in results)
        total_voice_violations = sum(r.voice_violations for r in results)

        print()
        print("=" * 60)
        print("  鲁棒性评估汇总报告")
        print("=" * 60)
        print(f"  总测例数:      {total}")
        print(f"  ✅ PASS:       {passed} ({passed/total*100:.0f}%)")
        print(f"  ⚠️  WARN:       {warned} ({warned/total*100:.0f}%)")
        print(f"  ❌ FAIL:       {failed} ({failed/total*100:.0f}%)")
        print(f"  💥 ERROR:      {errors}")
        print()
        print(f"  平均幻觉得分:  {avg_h:.2f}")
        print(f"  平均一致性:    {avg_c:.2f}")
        print(f"  平均综合评分:  {avg_auto:.2f}")
        print(f"  平均耗时:      {avg_time:.0f}ms")
        print()
        print(f"  幻觉标记总数:  {total_h_flags}")
        print(f"  时间违规总数:  {total_time_violations}")
        print(f"  口吻违规总数:  {total_voice_violations}")
        print()

        # 按类别统计
        print("  --- 按类别统计 ---")
        by_category = {}
        for r in results:
            cat = r.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "passed": 0, "failed": 0, "scores": []}
            by_category[cat]["total"] += 1
            by_category[cat]["scores"].append(r.auto_score)
            if r.auto_grade == "PASS":
                by_category[cat]["passed"] += 1
            elif r.auto_grade == "FAIL":
                by_category[cat]["failed"] += 1

        for cat, stats in sorted(by_category.items()):
            avg = sum(stats["scores"]) / len(stats["scores"])
            print(f"  {cat}: {stats['total']}条 | "
                  f"PASS={stats['passed']} FAIL={stats['failed']} | "
                  f"均分={avg:.2f}")

        # 按严重度统计
        print()
        print("  --- 按严重度统计 ---")
        by_severity = {}
        for r in results:
            sev = r.severity
            if sev not in by_severity:
                by_severity[sev] = {"total": 0, "passed": 0, "scores": []}
            by_severity[sev]["total"] += 1
            by_severity[sev]["scores"].append(r.auto_score)
            if r.auto_grade == "PASS":
                by_severity[sev]["passed"] += 1

        for sev in ["high", "medium", "low"]:
            if sev in by_severity:
                stats = by_severity[sev]
                avg = sum(stats["scores"]) / len(stats["scores"])
                print(f"  {sev}: {stats['total']}条 | "
                      f"PASS={stats['passed']}/{stats['total']} | "
                      f"均分={avg:.2f}")

        # 最差测例
        print()
        print("  --- 需要关注的测例 (auto_score < 0.6) ---")
        poor = [r for r in results if r.auto_score < 0.6]
        poor.sort(key=lambda x: x.auto_score)
        for r in poor[:10]:
            print(f"  [{r.case_id}] {r.question[:50]}... → {r.auto_score} [{r.auto_grade}]")
            if r.hallucination_flags:
                print(f"       幻觉: {r.hallucination_summary}")
            if r.time_violations or r.voice_violations:
                print(f"       一致性: {r.consistency_summary}")

        print()
        print("=" * 60)

    def generate_markdown_report(self, results: List[EvalResult], output_path: str):
        """生成 Markdown 格式的评估报告"""
        total = len(results)
        if total == 0:
            return

        passed = sum(1 for r in results if r.auto_grade == "PASS")
        warned = sum(1 for r in results if r.auto_grade == "WARN")
        failed = sum(1 for r in results if r.auto_grade == "FAIL")
        errors = sum(1 for r in results if r.error)
        avg_h = sum(r.hallucination_score for r in results) / total
        avg_c = sum(r.consistency_score for r in results) / total
        avg_auto = sum(r.auto_score for r in results) / total

        lines = []
        lines.append("# 鲁棒性自动化评估报告")
        lines.append("")
        lines.append(f"**评估时间**: {time.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**测例总数**: {total}")
        lines.append("")
        lines.append("## 总览")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| ✅ PASS | {passed} ({passed/total*100:.0f}%) |")
        lines.append(f"| ⚠️ WARN | {warned} ({warned/total*100:.0f}%) |")
        lines.append(f"| ❌ FAIL | {failed} ({failed/total*100:.0f}%) |")
        lines.append(f"| 💥 ERROR | {errors} |")
        lines.append(f"| 平均幻觉得分 | {avg_h:.2f} |")
        lines.append(f"| 平均一致性得分 | {avg_c:.2f} |")
        lines.append(f"| 平均综合评分 | {avg_auto:.2f} |")
        lines.append("")

        # 按类别
        lines.append("## 按类别统计")
        lines.append("")
        lines.append("| 类别 | 数量 | PASS | FAIL | 均分 |")
        lines.append("|------|------|------|------|------|")
        by_category = {}
        for r in results:
            cat = r.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "passed": 0, "failed": 0, "scores": []}
            by_category[cat]["total"] += 1
            by_category[cat]["scores"].append(r.auto_score)
            if r.auto_grade == "PASS":
                by_category[cat]["passed"] += 1
            elif r.auto_grade == "FAIL":
                by_category[cat]["failed"] += 1
        for cat, stats in sorted(by_category.items()):
            avg = sum(stats["scores"]) / len(stats["scores"])
            lines.append(f"| {cat} | {stats['total']} | {stats['passed']} | {stats['failed']} | {avg:.2f} |")
        lines.append("")

        # 需要关注的
        lines.append("## 需要关注的测例 (auto_score < 0.6)")
        lines.append("")
        poor = [r for r in results if r.auto_score < 0.6]
        poor.sort(key=lambda x: x.auto_score)
        if poor:
            lines.append("| ID | 问题 | 评分 | 幻觉 | 一致性 | 判定 |")
            lines.append("|----|------|------|------|--------|------|")
            for r in poor:
                lines.append(
                    f"| {r.case_id} | {r.question[:40]}... | {r.auto_score} | "
                    f"{r.hallucination_score} | {r.consistency_score} | {r.auto_grade} |"
                )
        else:
            lines.append("无")
        lines.append("")

        # 各测例详情表
        lines.append("## 全部测例详情")
        lines.append("")
        lines.append("| ID | 类别 | 严重度 | 综合评分 | 幻觉 | 一致性 | 判定 |")
        lines.append("|----|------|--------|----------|------|--------|------|")
        for r in results:
            lines.append(
                f"| {r.case_id} | {r.category} | {r.severity} | "
                f"{r.auto_score} | {r.hallucination_score} | {r.consistency_score} | "
                f"{'✅' if r.auto_grade == 'PASS' else '⚠️' if r.auto_grade == 'WARN' else '❌'} |"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"Markdown 报告已保存至: {output_path}")


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="鲁迅数字人 鲁棒性自动化评估脚本"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="CSV 输出路径 (默认: docs/eval-results-YYYYMMDD-HHMM.csv)",
    )
    parser.add_argument(
        "--category", "-c", type=str, default=None,
        choices=["hallucination_induction", "time_boundary", "knowledge_gap"],
        help="只评估指定类别",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="只评估前 N 条",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="显示每条详情",
    )
    parser.add_argument(
        "--markdown", "-m", type=str, default=None,
        help="同时生成 Markdown 报告",
    )

    args = parser.parse_args()

    # 输出路径
    if args.output:
        csv_path = args.output
    else:
        timestamp = time.strftime("%Y%m%d-%H%M")
        csv_path = os.path.join(ROOT_DIR, "docs", f"eval-results-{timestamp}.csv")

    # 确保 docs 目录存在
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # 执行评估
    runner = EvalRunner(verbose=args.verbose)
    results = runner.run_all(
        category=args.category,
        limit=args.limit,
    )

    # 保存
    if results:
        runner.save_results(results, csv_path)
        runner.print_summary(results)

        if args.markdown:
            runner.generate_markdown_report(results, args.markdown)
        else:
            # 默认也生成 markdown
            md_path = csv_path.replace(".csv", ".md")
            runner.generate_markdown_report(results, md_path)


if __name__ == "__main__":
    main()
