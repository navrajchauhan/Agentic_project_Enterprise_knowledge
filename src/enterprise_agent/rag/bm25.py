"""BM25 (lexical) retrieval, implemented from scratch.

Why not ``rank_bm25``?  The dependency is tiny, but writing BM25 explicitly
makes the lexical half of the hybrid retriever inspectable - you can print
per-term contributions when a ranking looks wrong.  It is ~80 lines of
well-understood maths.

Scoring formula (Okapi BM25):

.. math::

    \\text{score}(q, d) = \\sum_{t \\in q} IDF(t) \\cdot
        \\frac{f(t,d)\\,(k_1 + 1)}{f(t,d) + k_1 \\left(1 - b + b \\frac{|d|}{\\text{avgdl}}\\right)}

Two practical details:

* **IDF is always positive.**  The classic ``log((N - n + 0.5) / (n + 0.5))``
  goes negative for terms that appear in more than half the documents, which
  makes common terms *penalise* a document.  The ``log(1 + ...)`` variant used
  here (the one Lucene/Elasticsearch rely on) avoids that footgun.
* **Inverted index.**  Only documents containing at least one query term are
  visited, so scoring is proportional to posting-list length rather than to
  corpus size.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from ..utils.text import tokenize


class BM25Index:
    """In-memory BM25 index over a tokenised corpus."""

    def __init__(
        self,
        tokenised_corpus: list[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not tokenised_corpus:
            raise ValueError("BM25Index needs at least one document")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b
        self.n_docs = len(tokenised_corpus)
        self.doc_lengths: list[int] = [len(doc) for doc in tokenised_corpus]
        self.avg_doc_length = sum(self.doc_lengths) / self.n_docs or 1.0

        # postings[term] -> list of (doc_index, term_frequency)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_freq: Counter[str] = Counter()

        for doc_index, tokens in enumerate(tokenised_corpus):
            term_frequencies = Counter(tokens)
            for term, frequency in term_frequencies.items():
                postings[term].append((doc_index, frequency))
                doc_freq[term] += 1

        self._postings = dict(postings)
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    # -- scoring -----------------------------------------------------------
    def idf(self, term: str) -> float:
        """Inverse document frequency; 0 for unseen terms."""
        return self._idf.get(term, 0.0)

    def scores(self, query: str | list[str]) -> list[float]:
        """Score every document against the query (unseen terms contribute 0)."""
        tokens = tokenize(query) if isinstance(query, str) else query
        scores = [0.0] * self.n_docs
        if not tokens:
            return scores

        for term in set(tokens):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for doc_index, frequency in postings:
                length_norm = 1.0 - self.b + self.b * (
                    self.doc_lengths[doc_index] / self.avg_doc_length
                )
                denominator = frequency + self.k1 * length_norm
                scores[doc_index] += idf * (frequency * (self.k1 + 1.0)) / denominator
        return scores

    def search(self, query: str | list[str], top_k: int = 50) -> list[tuple[int, float]]:
        """Return ``(doc_index, score)`` pairs, best first, zero scores dropped.

        Dropping zeros matters for the fusion step: a document that merely
        exists must not occupy a rank.
        """
        scored = self.scores(query)
        ranked = [(index, score) for index, score in enumerate(scored) if score > 0.0]
        ranked.sort(key=lambda item: (-item[1], item[0]))  # deterministic ties
        return ranked[:top_k]

    def explain(self, query: str, doc_index: int, top_terms: int = 5) -> list[tuple[str, float]]:
        """Per-term contribution breakdown for one document.

        Handy when a ranking looks wrong: the usual causes are a missing term
        (tokenisation) or a saturated term (every document matches it).
        """
        tokens = tokenize(query)
        length_norm = 1.0 - self.b + self.b * (self.doc_lengths[doc_index] / self.avg_doc_length)
        contributions: list[tuple[str, float]] = []
        postings_by_term = {term: dict(post) for term, post in self._postings.items()}

        for term in set(tokens):
            postings = postings_by_term.get(term, {})
            frequency = postings.get(doc_index, 0)
            if frequency == 0:
                continue
            idf = self._idf[term]
            contribution = idf * (frequency * (self.k1 + 1.0)) / (
                frequency + self.k1 * length_norm
            )
            contributions.append((term, contribution))

        contributions.sort(key=lambda item: -item[1])
        return contributions[:top_terms]
