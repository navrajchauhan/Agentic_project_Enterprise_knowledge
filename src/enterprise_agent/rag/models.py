"""Core data structures for the retrieval pipeline.

Plain dataclasses (not pydantic) because these objects sit in hot loops - every
query scores hundreds of chunks - and because they need to be trivially
serialisable to JSON for the persisted index.

Two objects matter most:

``Chunk``
    The atomic unit of retrieval.  It carries the citation metadata, so the
    answer generator can point at "hr_policy.md - Annual Leave" instead of a
    faceless vector id.

``ScoredChunk``
    A chunk plus the per-retriever scores that produced it.  Keeping BM25 and
    dense scores separate (instead of only the fused score) is what makes the
    hybrid ranking debuggable and is what the evaluation harness reports on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A source document loaded from the knowledge base."""

    doc_id: str  # stable id, derived from the file name (e.g. "hr_policy")
    title: str  # human title, from the H1 heading when present
    source_path: str  # path relative to the knowledge base root
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of a document."""

    chunk_id: str  # globally unique, e.g. "hr_policy::0007"
    doc_id: str
    title: str
    section: str  # section heading path, e.g. "Annual Leave - Carry Over"
    text: str
    chunk_index: int  # position of this chunk inside its document
    word_start: int  # offset of the chunk inside its section (word index)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        """Compact human-readable source label used in answers/reports."""
        section = self.section.strip()
        return f"{self.doc_id}.md - {section}" if section else f"{self.doc_id}.md"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "word_start": self.word_start,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=payload["chunk_id"],
            doc_id=payload["doc_id"],
            title=payload.get("title", ""),
            section=payload.get("section", ""),
            text=payload["text"],
            chunk_index=int(payload.get("chunk_index", 0)),
            word_start=int(payload.get("word_start", 0)),
            metadata=payload.get("metadata", {}) or {},
        )


@dataclass
class ScoredChunk:
    """A retrieval hit with the score breakdown that produced it."""

    chunk: Chunk
    score: float = 0.0  # fused (RRF) score used for ordering
    bm25_score: float = 0.0
    dense_score: float = 0.0
    bm25_rank: int | None = None  # 1-based rank in the lexical ranking
    dense_rank: int | None = None  # 1-based rank in the vector ranking
    expanded_text: str = ""  # hit text with neighbouring chunks stitched on

    @property
    def context_text(self) -> str:
        """Text handed to the LLM: the expanded window when available."""
        return self.expanded_text or self.chunk.text

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "section": self.chunk.section,
            "chunk_index": self.chunk.chunk_index,
            "score": round(self.score, 6),
            "bm25_score": round(self.bm25_score, 6),
            "dense_score": round(self.dense_score, 6),
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
            "citation": self.chunk.citation_label,
        }
        if include_text:
            payload["text"] = self.context_text
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoredChunk:
        """Rebuild a hit from the serialised form stored in agent state."""
        chunk = Chunk(
            chunk_id=payload["chunk_id"],
            doc_id=payload["doc_id"],
            title=payload.get("title", ""),
            section=payload.get("section", ""),
            text=payload.get("text", ""),
            chunk_index=int(payload.get("chunk_index", 0)),
            word_start=0,
        )
        return cls(
            chunk=chunk,
            score=float(payload.get("score", 0.0)),
            bm25_score=float(payload.get("bm25_score", 0.0)),
            dense_score=float(payload.get("dense_score", 0.0)),
            bm25_rank=payload.get("bm25_rank"),
            dense_rank=payload.get("dense_rank"),
            expanded_text=payload.get("text", ""),
        )


@dataclass(frozen=True)
class Citation:
    """A validated reference from an answer back to a retrieved chunk."""

    marker: int  # the [n] the model wrote
    chunk_id: str
    doc_id: str
    title: str
    section: str

    @property
    def label(self) -> str:
        section = self.section.strip()
        return f"{self.doc_id}.md - {section}" if section else f"{self.doc_id}.md"

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "label": self.label,
        }
