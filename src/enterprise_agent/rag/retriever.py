"""Hybrid retrieval: BM25 + dense vectors fused with Reciprocal Rank Fusion.

Why hybrid at all?
    Lexical (BM25) search nails exact enterprise vocabulary - "P1", "99.9%",
    "BitLocker", "USD 180".  Dense search handles the cases where the user
    words it differently ("how much can I claim for a hotel?") and absorbs
    typos/inflections.  Neither alone is good enough for policy QA, and the two
    fail on *different* queries, which is exactly when fusion pays off.

Why RRF instead of adding the scores?
    BM25 scores are unbounded and corpus-dependent; cosine similarities live in
    a narrow band.  Adding them requires fragile per-corpus normalisation.  RRF
    only looks at *ranks*, so it needs no calibration:

    .. math:: score(d) = \\sum_i \\frac{w_i}{k + rank_i(d)}

    with ``k = 60`` (the value from the original Cormack et al. paper).  The
    weighted variant below lets you bias towards lexical or semantic search
    without changing the code.

Post-processing that measurably helps enterprise QA:

* **Neighbour expansion** - prepend/append the adjacent chunk of a hit so a hit
  that begins mid-sentence is still readable by the LLM.
* **Near-duplicate suppression** - overlapping windows can produce two hits
  with almost the same text; spending two context slots on them wastes budget.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..config import Settings
from ..logging_setup import get_logger
from ..utils.text import content_tokens, truncate
from .embeddings import build_embedder, embedding_text
from .indexer import HybridIndex
from .models import Chunk, ScoredChunk

logger = get_logger(__name__)


class HybridRetriever:
    """Query-time retriever over a built ``HybridIndex``."""

    #: Fetch this multiple of ``top_k`` from each retriever before fusing, so
    #: that a hit ranked 3rd by BM25 and 2nd by the dense model still fuses to
    #: the top.  Too small and fusion cannot reorder; too large wastes cycles.
    CANDIDATE_MULTIPLIER = 5
    MAX_CANDIDATES = 100

    #: Jaccard similarity above which two hits are considered duplicates.
    DUPLICATE_THRESHOLD = 0.9

    def __init__(self, index: HybridIndex, settings: Settings) -> None:
        self.index = index
        self.settings = settings
        self.embedder = build_embedder(settings)
        # The embedder must match the one that built the index, otherwise the
        # query vector and the matrix live in different spaces.
        if self.embedder.name != index.embedder_name:
            raise ValueError(
                f"Retriever embedder {self.embedder.name!r} does not match index "
                f"embedder {index.embedder_name!r}; rebuild the index."
            )

    # -- public API --------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        doc_ids: Iterable[str] | None = None,
    ) -> list[ScoredChunk]:
        """Return fused, deduplicated hits for ``query``.

        Parameters
        ----------
        query
            Natural-language query.  Empty or whitespace-only returns ``[]``.
        top_k
            Number of hits to return (defaults to ``TOP_K``).
        doc_ids
            Optional document allow-list, e.g. ``["hr_policy"]``.  Filtering
            happens before fusion so the ranking is not distorted by documents
            the caller excluded.
        """
        query = (query or "").strip()
        if not query:
            return []

        limit = top_k or self.settings.top_k
        candidates = min(self.MAX_CANDIDATES, max(limit * self.CANDIDATE_MULTIPLIER, limit))
        allowed = set(doc_ids) if doc_ids else None

        lexical = self._lexical_ranking(query, candidates, allowed)
        semantic = self._dense_ranking(query, candidates, allowed)
        fused = self._fuse(lexical, semantic)

        hits = self._with_expansion(fused)
        hits = self._drop_near_duplicates(hits, limit)
        return hits[:limit]

    def context_block(self, hits: list[ScoredChunk], *, max_chars: int | None = None) -> str:
        """Render hits as a numbered evidence block for the LLM.

        The numbering *is* the citation contract: the generator is told it may
        only cite ``[1]..[n]`` from this block, and the output guard validates
        every marker against the same list, so an invented source cannot
        survive to the user.

        Each hit is capped at a share of the budget (``budget // 4``) so that one
        long neighbour-expanded hit cannot starve the others: three medium
        passages are worth more to the model than one huge one.
        """
        budget = max_chars or self.settings.max_context_chars
        per_hit_cap = max(300, budget // 4)
        blocks: list[str] = []
        used = 0

        for position, hit in enumerate(hits, start=1):
            text = " ".join(hit.context_text.split())
            header = (
                f"[{position}] source: {hit.chunk.citation_label} "
                f"(chunk_id={hit.chunk.chunk_id}, fused_score={hit.score:.4f})"
            )
            remaining = budget - used - len(header)
            if remaining < 120:  # not enough room for a useful quote
                logger.debug("Context budget exhausted after %d hit(s)", position - 1)
                break
            body = truncate(text, min(remaining, per_hit_cap))
            block = f"{header}\n{body}"
            blocks.append(block)
            used += len(block) + 2  # +2 for the blank line separator

        return "\n\n".join(blocks)

    # -- individual retrievers --------------------------------------------
    def _lexical_ranking(
        self, query: str, limit: int, allowed: set[str] | None
    ) -> list[tuple[int, float]]:
        """BM25 ranking, as ``(chunk_position, score)``."""
        ranked = self.index.bm25.search(query, top_k=self.MAX_CANDIDATES)
        filtered = [
            (position, score)
            for position, score in ranked
            if allowed is None or self.index.chunks[position].doc_id in allowed
        ]
        return filtered[:limit]

    def _dense_ranking(
        self, query: str, limit: int, allowed: set[str] | None
    ) -> list[tuple[int, float]]:
        """Vector ranking, translated into chunk positions.

        ``VectorStore.search`` returns ids; the fusion step needs positions, so
        the ids are mapped back through a small lookup built once per query.
        """
        query_vector = self.embedder.encode([query])[0]
        # Over-fetch when a filter is active, since filtering can drop many hits.
        fetch = self.MAX_CANDIDATES * 2 if allowed else self.MAX_CANDIDATES
        hits = self.index.vectors.search(query_vector, top_k=fetch)

        positions: list[tuple[int, float]] = []
        for chunk_id, score in hits:
            chunk = self.index.get(chunk_id)
            if chunk is None:  # pragma: no cover - index/vector mismatch
                continue
            if allowed is not None and chunk.doc_id not in allowed:
                continue
            positions.append((self._position_of(chunk), score))
            if len(positions) >= limit:
                break
        return positions

    def _position_of(self, chunk: Chunk) -> int:
        """Index of ``chunk`` inside ``index.chunks`` (memoised)."""
        cache = getattr(self, "_position_cache", None)
        if cache is None or cache[0] is not self.index.chunks:
            cache = (self.index.chunks, {c.chunk_id: i for i, c in enumerate(self.index.chunks)})
            self._position_cache = cache
        return cache[1][chunk.chunk_id]

    # -- fusion ------------------------------------------------------------
    def _fuse(
        self,
        lexical: list[tuple[int, float]],
        semantic: list[tuple[int, float]],
    ) -> list[ScoredChunk]:
        """Weighted Reciprocal Rank Fusion of the two rankings."""
        total_weight = self.settings.bm25_weight + self.settings.dense_weight
        w_lexical = self.settings.bm25_weight / total_weight
        w_semantic = self.settings.dense_weight / total_weight
        k = self.settings.rrf_k

        fused: dict[int, ScoredChunk] = {}

        for rank, (position, score) in enumerate(lexical, start=1):
            chunk = self.index.chunks[position]
            hit = fused.setdefault(position, ScoredChunk(chunk=chunk))
            hit.bm25_score = score
            hit.bm25_rank = rank
            hit.score += w_lexical / (k + rank)

        for rank, (position, score) in enumerate(semantic, start=1):
            chunk = self.index.chunks[position]
            hit = fused.setdefault(position, ScoredChunk(chunk=chunk))
            hit.dense_score = score
            hit.dense_rank = rank
            hit.score += w_semantic / (k + rank)

        # Sort by fused score; break ties deterministically by document order so
        # repeated runs of the same query return the same ordering.
        return sorted(
            fused.values(),
            key=lambda hit: (-hit.score, hit.chunk.doc_id, hit.chunk.chunk_index),
        )

    # -- post-processing ---------------------------------------------------
    def _with_expansion(self, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        """Stitch neighbouring chunks onto each hit for readable context."""
        if not self.settings.enable_neighbor_expansion:
            return hits

        for hit in hits:
            siblings = self.index.neighbours(hit.chunk, radius=1)
            if not siblings:
                hit.expanded_text = hit.chunk.text
                continue
            # Keep document order: previous chunk(s) before, next after.
            ordered = sorted(
                [*siblings, hit.chunk],
                key=lambda chunk: chunk.chunk_index,
            )
            hit.expanded_text = "\n\n".join(chunk.text for chunk in ordered)
        return hits

    def _drop_near_duplicates(self, hits: list[ScoredChunk], limit: int) -> list[ScoredChunk]:
        """Remove hits whose text largely repeats a better-ranked hit.

        Uses Jaccard similarity over content tokens rather than a vector
        comparison: it is cheap, has no model dependency, and overlapping
        windows are near-identical *lexically* by construction.
        """
        kept: list[ScoredChunk] = []
        kept_tokens: list[set[str]] = []

        for hit in hits:
            tokens = content_tokens(hit.context_text)
            duplicate = False
            for existing in kept_tokens:
                union = tokens | existing
                if not union:
                    continue
                if len(tokens & existing) / len(union) >= self.DUPLICATE_THRESHOLD:
                    duplicate = True
                    break
            if duplicate:
                logger.debug("Dropping near-duplicate hit %s", hit.chunk.chunk_id)
                continue
            kept.append(hit)
            kept_tokens.append(tokens)
            if len(kept) >= limit:
                break

        return kept

    # -- introspection -----------------------------------------------------
    def explain(self, query: str, top_k: int = 3) -> str:
        """Human-readable ranking breakdown (used by ``run.py ask --explain``)."""
        hits = self.search(query, top_k=top_k)
        lines = [f"query: {query}", f"index: {len(self.index)} chunks, embedder: {self.index.embedder_name}"]
        for rank, hit in enumerate(hits, start=1):
            lines.append(
                f"  {rank}. {hit.chunk.chunk_id} score={hit.score:.5f} "
                f"bm25={hit.bm25_score:.3f}@{hit.bm25_rank} "
                f"dense={hit.dense_score:.3f}@{hit.dense_rank} "
                f"| {hit.chunk.citation_label}"
            )
        if not hits:
            lines.append("  (no hits)")
        return "\n".join(lines)


def scoring_text_for(chunk: Chunk) -> str:
    """Convenience re-export so callers do not import ``embeddings`` directly."""
    return embedding_text(chunk.title, chunk.section, chunk.text)
