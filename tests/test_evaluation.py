"""Evaluation harness: metrics, golden-set execution and reporting."""

from __future__ import annotations

import json

import pytest

from enterprise_agent.evaluation.retrieval_metrics import (
    aggregate,
    evaluate_ranking,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from enterprise_agent.evaluation.runner import EvalCase, EvaluationRunner, load_dataset


# ---------------------------------------------------------------------------
# Metric maths
# ---------------------------------------------------------------------------


def test_hit_at_k() -> None:
    assert hit_at_k(["a", "b"], {"b"}, k=2) == 1.0
    assert hit_at_k(["a", "b"], {"b"}, k=1) == 0.0
    assert hit_at_k([], {"b"}, k=5) == 0.0


def test_recall_at_k() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0
    assert recall_at_k(["a", "b"], {"a", "c"}, k=2) == pytest.approx(0.5)


def test_precision_at_k() -> None:
    assert precision_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(2 / 3)
    assert precision_at_k([], {"a"}, k=3) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert reciprocal_rank(["a"], {"a"}) == 1.0


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)


def test_ndcg_penalises_late_hits() -> None:
    early = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
    late = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
    assert early > late


def test_evaluate_ranking_and_aggregate() -> None:
    first = evaluate_ranking(["a", "b"], {"a"}, k=5)
    second = evaluate_ranking(["x"], {"x"}, k=5)
    summary = aggregate([first, second])

    assert set(summary) == {"hit", "recall", "precision", "mrr", "ndcg"}
    assert summary["hit"] == 1.0
    assert summary["recall"] == 1.0


# ---------------------------------------------------------------------------
# Dataset + end-to-end harness
# ---------------------------------------------------------------------------


def test_golden_set_loads_and_is_consistent(project_root) -> None:
    cases = load_dataset(project_root / "evals" / "golden_set.json")
    assert len(cases) >= 12

    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids)), "case ids must be unique"

    for case in cases:
        assert case.question.strip()
        if case.kind == "retrieval" and not case.should_abstain:
            assert case.relevant_docs, f"{case.id} needs relevant_docs"
        if case.kind == "tool":
            assert case.expect_tool, f"{case.id} needs expect_tool"

    # Exactly one case must demand an abstention (the out-of-scope question).
    assert sum(1 for case in cases if case.should_abstain) == 1


def test_eval_case_parsing() -> None:
    case = EvalCase.from_dict(
        {"id": "x", "question": "q", "relevant_docs": ["doc"], "answer_keywords": ["1"]}
    )
    assert case.kind == "retrieval"
    assert case.relevant_docs == ["doc"]
    assert case.should_abstain is False


def test_evaluation_runner_retrieval_only(settings, runner) -> None:
    evaluator = EvaluationRunner(settings, agent=runner, judge=None)
    report = evaluator.run(retrieval_only=True, limit=6)

    assert report["retrieval_only"] is True
    assert report["case_count"] == 6
    retrieval = report["retrieval"]
    assert set(retrieval) == {f"hit@{settings.eval_top_k}", f"recall@{settings.eval_top_k}",
                              f"precision@{settings.eval_top_k}", "mrr", f"ndcg@{settings.eval_top_k}"}
    # The bundled knowledge base should be easy to retrieve from.
    assert retrieval[f"hit@{settings.eval_top_k}"] >= 0.8


def test_evaluation_runner_full_run_writes_reports(settings, runner) -> None:
    evaluator = EvaluationRunner(settings, agent=runner, judge=None)
    report = evaluator.run(retrieval_only=False, limit=4)

    assert report["answer"]["keyword_coverage"] >= 0.5
    assert report["answer"]["abstention_accuracy"] >= 0.75

    json_path, markdown_path = evaluator.write_report(report)
    assert json_path.exists() and markdown_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["case_count"] == 4
    assert len(payload["cases"]) == 4

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Evaluation report" in markdown
    assert "## Per-case results" in markdown


def test_evaluation_handles_a_failing_case(settings, runner, monkeypatch) -> None:
    """One broken case must not abort the run."""
    evaluator = EvaluationRunner(settings, agent=runner, judge=None)

    original_ask = runner.ask

    def flaky_ask(question: str, *, thread_id: str = "default"):
        if "annual leave" in question:
            raise RuntimeError("simulated failure")
        return original_ask(question, thread_id=thread_id)

    monkeypatch.setattr(runner, "ask", flaky_ask)
    report = evaluator.run(retrieval_only=False, limit=3)

    failed = [case for case in report["cases"] if case["error"]]
    assert failed, "the failing case must be reported, not raised"
    assert "simulated failure" in failed[0]["error"]
