"""Retrieval metrics, implemented without external dependencies.

All metrics treat a case's ``relevant_docs`` as the ground-truth set and the
ranked ``retrieved_docs`` as the prediction.  Document-level relevance is used
because that is what a citation can be validated against: an answer that cites
the wrong document is wrong even if the chunk happened to be adjacent.

Metrics
-------
``hit@k``        did at least one relevant document appear in the top k?
``recall@k``     share of relevant documents found in the top k
``precision@k``  share of the top k that are relevant
``mrr``          1 / rank of the first relevant document (0 if none)
``ndcg@k``       rank-weighted gain with binary relevance

Report ``hit@k`` and ``recall@k`` together: hit@k answers "can the agent find
*anything* useful", recall@k answers "does it find *all* the evidence needed" -
and multi-document questions need the latter.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    """Collapse duplicates while keeping the ranking order.

    Retrieval returns chunks, not documents, so several hits can belong to the
    same document; metric denominators must not be inflated by that.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    if not relevant or k <= 0:
        return 0.0
    gains = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved[:k]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_ranking(retrieved_docs: Sequence[str], relevant: set[str], k: int) -> dict[str, float]:
    """All metrics for one query."""
    ranked = dedupe_preserving_order(retrieved_docs)
    return {
        "hit": hit_at_k(ranked, relevant, k),
        "recall": recall_at_k(ranked, relevant, k),
        "precision": precision_at_k(ranked, relevant, k),
        "mrr": reciprocal_rank(ranked, relevant),
        "ndcg": ndcg_at_k(ranked, relevant, k),
    }


def aggregate(results: Sequence[dict[str, float]]) -> dict[str, float]:
    """Macro-average a list of per-query metric dictionaries."""
    if not results:
        return {"hit": 0.0, "recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}
    keys = results[0].keys()
    return {key: mean([result[key] for result in results]) for key in keys}
