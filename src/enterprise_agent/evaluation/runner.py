"""Evaluation harness.

Runs a golden set of questions through the real pipeline twice over:

1. **Retrieval only** - hybrid search, scored with hit@k / recall@k / MRR / nDCG.
   This is the number to watch when tuning chunking, weights or the embedder,
   and it needs no API key.
2. **End to end** - the full agent, then per-answer scoring:
   keyword coverage, citation accuracy, abstention correctness, the output
   guard's grounding score, and (optionally) LLM-as-a-judge faithfulness.

Why both?
    They fail differently.  A retrieval regression shows up in recall@k; a
    prompt regression shows up in faithfulness with retrieval unchanged.  Having
    the two separate means a bad change tells you *which half* broke.

Reports are written as JSON (machine-readable, for CI) and Markdown (for humans).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..guardrails.output_guard import is_abstention
from ..logging_setup import get_logger
from .llm_judge import LLMJudge
from .retrieval_metrics import aggregate, dedupe_preserving_order, evaluate_ranking

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """One golden-set question."""

    id: str
    question: str
    kind: str = "retrieval"  # "retrieval" | "tool"
    relevant_docs: list[str] = field(default_factory=list)
    answer_keywords: list[str] = field(default_factory=list)
    should_abstain: bool = False
    expect_tool: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalCase:
        return cls(
            id=payload["id"],
            question=payload["question"],
            kind=payload.get("kind", "retrieval"),
            relevant_docs=payload.get("relevant_docs", []) or [],
            answer_keywords=payload.get("answer_keywords", []) or [],
            should_abstain=bool(payload.get("should_abstain", False)),
            expect_tool=payload.get("expect_tool"),
            notes=payload.get("notes", ""),
        )


def load_dataset(path: Path) -> list[EvalCase]:
    """Load and validate the golden set."""
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}. "
            "Point EVAL_DATASET at a JSON file, or use the bundled evals/golden_set.json."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", payload if isinstance(payload, list) else [])
    cases = [EvalCase.from_dict(item) for item in raw_cases]
    if not cases:
        raise ValueError(f"No cases found in {path}")
    logger.info("Loaded %d evaluation case(s) from %s", len(cases), path.name)
    return cases


# ---------------------------------------------------------------------------
# Report objects
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    id: str
    question: str
    kind: str
    retrieved_docs: list[str] = field(default_factory=list)
    retrieval: dict[str, float] = field(default_factory=dict)
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    keyword_coverage: float = 0.0
    missing_keywords: list[str] = field(default_factory=list)
    citation_accuracy: float | None = None
    abstained: bool = False
    abstention_correct: bool | None = None
    grounding: float = 0.0
    used_tools: list[str] = field(default_factory=list)
    tool_ok: bool | None = None
    judge: dict[str, Any] | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "kind": self.kind,
            "retrieved_docs": self.retrieved_docs,
            "retrieval": {key: round(value, 4) for key, value in self.retrieval.items()},
            "answer": self.answer,
            "citations": self.citations,
            "keyword_coverage": round(self.keyword_coverage, 3),
            "missing_keywords": self.missing_keywords,
            "citation_accuracy": None
            if self.citation_accuracy is None
            else round(self.citation_accuracy, 3),
            "abstained": self.abstained,
            "abstention_correct": self.abstention_correct,
            "grounding": round(self.grounding, 3),
            "used_tools": self.used_tools,
            "tool_ok": self.tool_ok,
            "judge": self.judge,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class EvaluationRunner:
    """Executes the golden set against a wired agent."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        agent: Any | None = None,
        judge: LLMJudge | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        if agent is None:
            # Import here: constructing the agent is expensive (index + models).
            from ..agent.runner import AgentRunner  # noqa: PLC0415

            agent = AgentRunner(self.settings)
        self.agent = agent

        self.judge = judge
        if judge is None and self.settings.enable_llm_judge:
            try:
                self.judge = LLMJudge(self.settings)
            except Exception as exc:  # noqa: BLE001 - judging is optional
                logger.warning("LLM judge unavailable: %s", exc)

    # -- main --------------------------------------------------------------
    def run(
        self,
        *,
        retrieval_only: bool = False,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate every case and return the full report dictionary."""
        cases = load_dataset(self.settings.eval_dataset_path)
        if limit:
            cases = cases[:limit]

        started = time.perf_counter()
        results: list[CaseResult] = []

        for index, case in enumerate(cases, start=1):
            logger.info("[%d/%d] %s", index, len(cases), case.id)
            try:
                results.append(self._run_case(case, retrieval_only=retrieval_only))
            except Exception as exc:  # noqa: BLE001 - one bad case must not stop the run
                logger.exception("Case %s failed", case.id)
                results.append(
                    CaseResult(id=case.id, question=case.question, kind=case.kind, error=str(exc))
                )

        report = self._aggregate(results, cases, elapsed=time.perf_counter() - started)
        report["retrieval_only"] = retrieval_only
        report["mode"] = getattr(self.agent, "mode", "unknown")
        return report

    # -- one case ----------------------------------------------------------
    def _run_case(self, case: EvalCase, *, retrieval_only: bool) -> CaseResult:
        started = time.perf_counter()
        result = CaseResult(id=case.id, question=case.question, kind=case.kind)
        relevant = set(case.relevant_docs)

        # 1. Retrieval metrics ------------------------------------------------
        hits = self.agent.retriever.search(case.question, top_k=self.settings.eval_top_k)
        retrieved_docs = dedupe_preserving_order([hit.chunk.doc_id for hit in hits])
        result.retrieved_docs = retrieved_docs
        if relevant:
            result.retrieval = evaluate_ranking(retrieved_docs, relevant, self.settings.eval_top_k)

        # 2. Full agent -------------------------------------------------------
        if retrieval_only:
            result.elapsed_seconds = time.perf_counter() - started
            return result

        answer = self.agent.ask(case.question, thread_id=f"eval-{case.id}")
        result.answer = answer.answer
        result.citations = answer.citations
        result.grounding = answer.confidence
        result.abstained = answer.abstained or is_abstention(answer.answer)
        result.used_tools = [call["tool"] for call in answer.tool_calls]

        # 3. Keyword coverage -------------------------------------------------
        lowered = answer.answer.lower()
        missing = [keyword for keyword in case.answer_keywords if keyword.lower() not in lowered]
        result.missing_keywords = missing
        if case.answer_keywords:
            result.keyword_coverage = 1.0 - (len(missing) / len(case.answer_keywords))

        # 4. Citation accuracy ------------------------------------------------
        if relevant and answer.citations:
            correct = sum(
                1 for citation in answer.citations if citation.get("doc_id") in relevant
            )
            result.citation_accuracy = correct / len(answer.citations)
        elif relevant and not answer.citations and not result.abstained:
            # A non-abstaining answer with no citations is wrong by definition.
            result.citation_accuracy = 0.0

        # 5. Abstention correctness -------------------------------------------
        result.abstention_correct = result.abstained == case.should_abstain

        # 6. Tool expectations ------------------------------------------------
        if case.expect_tool:
            result.tool_ok = case.expect_tool in result.used_tools

        # 7. Optional LLM judge -----------------------------------------------
        if self.judge is not None and case.kind == "retrieval" and not result.abstained:
            # Render the evidence in the same numbered shape the generator saw, so
            # the judge can map a claim to the passage [n] it cites.
            evidence = "\n\n".join(
                f"[{index}] source: {item.get('citation', item.get('doc_id', ''))} "
                f"(chunk_id={item.get('chunk_id', '?')}, "
                f"fused_score={float(item.get('score', 0.0)):.4f})\n"
                f"{item.get('text', '')}"
                for index, item in enumerate(answer.evidence, start=1)
            )
            scores = self.judge.score(
                question=case.question, answer=answer.answer, evidence_text=evidence
            )
            if scores is not None:
                result.judge = scores.to_dict()

        result.elapsed_seconds = time.perf_counter() - started
        return result

    # -- aggregation -------------------------------------------------------
    def _aggregate(
        self, results: list[CaseResult], cases: list[EvalCase], *, elapsed: float
    ) -> dict[str, Any]:
        retrieval_results = [result for result in results if result.retrieval]
        answer_results = [
            result for result in results if result.kind == "retrieval" and not result.error
        ]
        tool_results = [result for result in results if result.kind == "tool"]

        retrieval_summary = aggregate([result.retrieval for result in retrieval_results])
        retrieval_summary = {f"{key}@{self.settings.eval_top_k}" if key != "mrr" else "mrr": value
                             for key, value in retrieval_summary.items()}

        # Only cases that declare expected facts can be scored on fact coverage -
        # otherwise an abstention case (which correctly has no keywords) would be
        # averaged in as a zero and understate accuracy.
        cases_with_keywords = {case.id for case in cases if case.answer_keywords}
        keyword_scores = [
            result.keyword_coverage
            for result in answer_results
            if result.id in cases_with_keywords
        ]
        citation_scores = [
            result.citation_accuracy
            for result in answer_results
            if result.citation_accuracy is not None
        ]
        abstention_scores = [
            1.0 if result.abstention_correct else 0.0
            for result in answer_results
            if result.abstention_correct is not None
        ]
        grounding_scores = [result.grounding for result in answer_results]

        judge_results = [result.judge for result in answer_results if result.judge]
        judge_summary = {
            "faithfulness": _mean([item["faithfulness"] for item in judge_results]),
            "relevance": _mean([item["relevance"] for item in judge_results]),
            "citation_support": _mean([item["citation_support"] for item in judge_results]),
            "cases_scored": len(judge_results),
        }

        tool_scores = [1.0 if result.tool_ok else 0.0 for result in tool_results]

        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset": str(self.settings.eval_dataset_path.name),
            "top_k": self.settings.eval_top_k,
            "case_count": len(cases),
            "elapsed_seconds": round(elapsed, 2),
            "retrieval": {key: round(value, 4) for key, value in retrieval_summary.items()},
            "answer": {
                "keyword_coverage": round(_mean(keyword_scores), 3),
                "citation_accuracy": round(_mean(citation_scores), 3),
                "abstention_accuracy": round(_mean(abstention_scores), 3),
                "mean_grounding": round(_mean(grounding_scores), 3),
                "tool_success_rate": round(_mean(tool_scores), 3),
            },
            "judge": judge_summary,
            "cases": [result.to_dict() for result in results],
        }

    # -- persistence -------------------------------------------------------
    def write_report(self, report: dict[str, Any]) -> tuple[Path, Path]:
        """Write the JSON and Markdown reports; returns both paths."""
        directory = self.settings.report_path
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")

        json_path = directory / f"eval-{stamp}.json"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        markdown_path = directory / f"eval-{stamp}.md"
        markdown_path.write_text(render_markdown(report), encoding="utf-8")

        logger.info("Wrote evaluation reports to %s", directory)
        return json_path, markdown_path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    """Render a report as a human-readable Markdown document."""
    retrieval = report.get("retrieval", {})
    answer = report.get("answer", {})
    judge = report.get("judge", {})

    lines: list[str] = [
        "# Evaluation report",
        "",
        f"- Generated: {report.get('generated_at')}",
        f"- Dataset: `{report.get('dataset')}` ({report.get('case_count')} cases)",
        f"- Mode: **{report.get('mode')}**"
        + (" (retrieval only)" if report.get("retrieval_only") else ""),
        f"- Top-k: {report.get('top_k')}",
        f"- Wall clock: {report.get('elapsed_seconds')}s",
        "",
        "## Retrieval quality",
        "",
        "| Metric | Score |",
        "|---|---|",
    ]
    for key, value in retrieval.items():
        lines.append(f"| {key} | {value:.3f} |")

    lines += [
        "",
        "## Answer quality",
        "",
        "| Metric | Score |",
        "|---|---|",
    ]
    for key, value in answer.items():
        lines.append(f"| {key} | {value:.3f} |")

    if judge.get("cases_scored"):
        lines += [
            "",
            "## LLM-as-a-judge (1-5)",
            "",
            "| Axis | Score |",
            "|---|---|",
            f"| faithfulness | {judge['faithfulness']:.2f} |",
            f"| relevance | {judge['relevance']:.2f} |",
            f"| citation support | {judge['citation_support']:.2f} |",
            f"| cases scored | {judge['cases_scored']} |",
        ]

    lines += [
        "",
        "## Per-case results",
        "",
        "| Case | Hit@k | Recall@k | Keywords | Citations | Abstained | Grounding | Tools |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for case in report.get("cases", []):
        if case.get("error"):
            lines.append(f"| {case['id']} | error: {case['error'][:60]} | | | | | | |")
            continue

        retrieval = case.get("retrieval") or {}
        hit_cell = f"{retrieval.get('hit', 0):.0f}" if retrieval else "-"
        recall_cell = f"{retrieval.get('recall', 0):.2f}" if retrieval else "-"
        citation_cell = (
            f"{case['citation_accuracy']:.2f}" if case.get("citation_accuracy") is not None else "-"
        )
        tools_cell = ", ".join(case.get("used_tools") or []) or "-"
        lines.append(
            f"| {case['id']} | {hit_cell} | {recall_cell} | {case['keyword_coverage']:.2f} "
            f"| {citation_cell} | {'yes' if case['abstained'] else 'no'} "
            f"| {case['grounding']:.2f} | {tools_cell} |"
        )

    lines += ["", "## Notes", ""]
    abstained = [case["id"] for case in report.get("cases", []) if case.get("abstained")]
    if abstained:
        lines.append(f"- Abstained on: {', '.join(abstained)}")
    failed_tools = [
        case["id"] for case in report.get("cases", []) if case.get("tool_ok") is False
    ]
    if failed_tools:
        lines.append(f"- Tool expectation failed for: {', '.join(failed_tools)}")
    if not judge.get("cases_scored"):
        lines.append("- LLM judge was not run (offline mode, disabled, or no valid JSON).")

    lines.append("")
    return "\n".join(lines)
