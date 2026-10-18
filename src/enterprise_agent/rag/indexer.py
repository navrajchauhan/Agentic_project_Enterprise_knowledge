"""Building, persisting and loading the hybrid index.

The index is the join of three things:

==========================  ====================================================
``chunks.jsonl``            the chunk metadata that citations resolve to
``vectors.npy`` + ids       dense matrix for semantic search
in-memory ``BM25Index``     lexical statistics rebuilt from the chunks
==========================  ====================================================

``manifest.json`` records the embedding backend, the dimension and a hash of
every source file.  That gives us two safety properties cheaply:

* switching ``EMBEDDING_BACKEND`` cannot silently pair new queries with old
  vectors - loading raises instead;
* editing a document in the knowledge base makes the index *stale*, and
  ``Indexer.build()`` rebuilds it automatically.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..logging_setup import get_logger
from ..utils.text import stable_hash, tokenize
from .bm25 import BM25Index
from .chunking import chunk_documents
from .embeddings import Embedder, build_embedder, embedding_text
from .loaders import load_documents
from .models import Chunk
from .vector_store import VectorStore

logger = get_logger(__name__)

MANIFEST_VERSION = 1


@dataclass
class HybridIndex:
    """Everything needed to answer a query: chunks + lexical + dense."""

    chunks: list[Chunk]
    bm25: BM25Index
    vectors: VectorStore
    embedder_name: str
    manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Lookup structures for O(1) citation resolution and neighbour expansion.
        self._by_id: dict[str, Chunk] = {chunk.chunk_id: chunk for chunk in self.chunks}
        neighbours: dict[str, list[Chunk]] = {}
        for chunk in sorted(self.chunks, key=lambda item: (item.doc_id, item.chunk_index)):
            neighbours.setdefault(chunk.doc_id, []).append(chunk)
        self._by_doc: dict[str, list[Chunk]] = neighbours

    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def neighbours(self, chunk: Chunk, radius: int = 1, *, same_section: bool = True) -> list[Chunk]:
        """Chunks adjacent to ``chunk`` inside the same document.

        With ``same_section`` (the default) the window stops at a heading
        boundary.  That is deliberate: at a section boundary the text does *not*
        continue mid-sentence, so pulling the neighbouring section in only adds
        unrelated material - and, worse, it lets the model quote a sentence from
        a section other than the one the citation names.  Expansion exists for
        readability, not for widening the evidence.
        """
        siblings = self._by_doc.get(chunk.doc_id, [])
        low = max(0, chunk.chunk_index - radius)
        high = min(len(siblings), chunk.chunk_index + radius + 1)
        return [
            item
            for item in siblings[low:high]
            if item.chunk_id != chunk.chunk_id
            and (not same_section or item.section == chunk.section)
        ]

    @property
    def doc_ids(self) -> list[str]:
        return sorted(self._by_doc)


def _file_digest(path: Path) -> str:
    """Content hash of a source file (used for staleness detection)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class Indexer:
    """Builds the hybrid index from the knowledge base directory."""

    def __init__(self, settings: Settings, embedder: Embedder | None = None) -> None:
        self.settings = settings
        self.embedder = embedder or build_embedder(settings)

    # -- build -------------------------------------------------------------
    def build(self, *, force: bool = False) -> HybridIndex:
        """Return a fresh index, reusing the persisted one when still valid."""
        if not force and self.loaded_is_fresh():
            logger.info("Index is up to date; loading from %s", self.settings.index_path)
            return self.load()

        logger.info("Building hybrid index from %s", self.settings.knowledge_base_path)
        started = time.perf_counter()

        documents = load_documents(self.settings.knowledge_base_path)
        chunks = chunk_documents(
            documents,
            size_words=self.settings.chunk_size_words,
            overlap_words=self.settings.chunk_overlap_words,
        )

        # Scoring text = title + section + body (see embeddings.embedding_text).
        scoring_texts = [
            embedding_text(chunk.title, chunk.section, chunk.text) for chunk in chunks
        ]

        tokenised = [tokenize(text) for text in scoring_texts]
        bm25 = BM25Index(tokenised)

        vectors = VectorStore(self.embedder.dim)
        vectors.add([chunk.chunk_id for chunk in chunks], self.embedder.encode(scoring_texts))

        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "chunk_size_words": self.settings.chunk_size_words,
            "chunk_overlap_words": self.settings.chunk_overlap_words,
            "chunk_count": len(chunks),
            "document_count": len(documents),
            "documents": [
                {
                    "doc_id": document.doc_id,
                    "source_path": document.source_path,
                    "sha256": _file_digest(self.settings.knowledge_base_path / document.source_path),
                }
                for document in documents
            ],
            "build_seconds": round(time.perf_counter() - started, 3),
        }

        index = HybridIndex(
            chunks=chunks,
            bm25=bm25,
            vectors=vectors,
            embedder_name=self.embedder.name,
            manifest=manifest,
        )
        self.save(index)
        logger.info(
            "Indexed %d chunk(s) from %d document(s) in %.2fs",
            len(chunks),
            len(documents),
            manifest["build_seconds"],
        )
        return index

    # -- persistence -------------------------------------------------------
    def save(self, index: HybridIndex) -> None:
        directory = self.settings.index_path
        directory.mkdir(parents=True, exist_ok=True)
        index.vectors.save(directory)

        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in index.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        (directory / "manifest.json").write_text(
            json.dumps(index.manifest, indent=2), encoding="utf-8"
        )

    def load(self) -> HybridIndex:
        directory = self.settings.index_path
        manifest_path = directory / "manifest.json"
        chunks_path = directory / "chunks.jsonl"
        if not manifest_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"No index in {directory}. Run: python run.py ingest"
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("embedder") != self.embedder.name or manifest.get("dim") != self.embedder.dim:
            raise ValueError(
                f"Index was built with {manifest.get('embedder')} ({manifest.get('dim')}-d) "
                f"but the configured embedder is {self.embedder.name} ({self.embedder.dim}-d). "
                "Re-run ingest to rebuild it."
            )

        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        # BM25 statistics are cheap to rebuild, so they are not persisted: fewer
        # artefacts to keep in sync, and no risk of a stale postings list.
        tokenised = [tokenize(embedding_text(c.title, c.section, c.text)) for c in chunks]
        bm25 = BM25Index(tokenised)
        vectors = VectorStore.load(directory, self.embedder.dim)

        logger.info("Loaded index: %d chunk(s), embedder=%s", len(chunks), self.embedder.name)
        return HybridIndex(
            chunks=chunks,
            bm25=bm25,
            vectors=vectors,
            embedder_name=self.embedder.name,
            manifest=manifest,
        )

    # -- freshness ---------------------------------------------------------
    def loaded_is_fresh(self) -> bool:
        """True when a persisted index exists and matches the current sources."""
        manifest_path = self.settings.index_path / "manifest.json"
        if not manifest_path.exists():
            return False

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False

        if manifest.get("embedder") != self.embedder.name:
            return False
        if manifest.get("chunk_size_words") != self.settings.chunk_size_words:
            return False
        if manifest.get("chunk_overlap_words") != self.settings.chunk_overlap_words:
            return False

        recorded = {entry["source_path"]: entry["sha256"] for entry in manifest.get("documents", [])}
        knowledge_base = self.settings.knowledge_base_path
        if not knowledge_base.exists():
            return False

        current_paths = {
            path.name
            for path in knowledge_base.iterdir()
            if path.is_file() and path.suffix.lower() in {".md", ".markdown", ".txt", ".rst", ".pdf"}
        }
        if current_paths != set(recorded):
            return False

        try:
            return all(
                _file_digest(knowledge_base / name) == digest
                for name, digest in recorded.items()
            )
        except OSError:
            return False

    def fingerprint(self) -> str:
        """Short hash of the knowledge base state (useful for reporting)."""
        knowledge_base = self.settings.knowledge_base_path
        if not knowledge_base.exists():
            return "missing"
        digests = sorted(
            f"{path.name}:{_file_digest(path)}"
            for path in knowledge_base.iterdir()
            if path.is_file()
        )
        return stable_hash("|".join(digests))
