"""Offline evaluation of retrieval accuracy and hallucination control.

``retrieval_metrics``  hit@k, recall@k, precision@k, MRR, nDCG (no dependencies)
``llm_judge``          LLM-as-a-judge for faithfulness / relevance / citations
``runner``             golden-set execution, aggregation and reporting
"""

from __future__ import annotations

from .llm_judge import JudgeScores, LLMJudge
from .retrieval_metrics import (
    aggregate,
    evaluate_ranking,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .runner import CaseResult, EvalCase, EvaluationRunner, load_dataset

__all__ = [
    "CaseResult",
    "EvalCase",
    "EvaluationRunner",
    "JudgeScores",
    "LLMJudge",
    "aggregate",
    "evaluate_ranking",
    "hit_at_k",
    "load_dataset",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
