"""Embeddings.

Groq serves chat completions, not embeddings, so the dense half of the hybrid
retriever needs a local encoder.  Two backends are provided:

``hashing`` (default)
    Dependency-free **feature hashing** over word unigrams, word bigrams and
    character 4-grams, signed-hashed into a fixed-size vector and L2-normalised.
    It is deterministic, needs no model download and works offline - which
    keeps this project runnable and testable on any machine.  It behaves like a
    smoothed lexical matcher: strong on morphology and typos ("reimbursment"),
    weak on true synonymy.  BM25 covers exact terms, this covers the fuzzy
    overlaps, and the two are fused.

``sentence-transformers``
    Real semantic embeddings (default ``all-MiniLM-L6-v2``).  Enable it with
    ``EMBEDDING_BACKEND=sentence-transformers`` after
    ``pip install sentence-transformers``.  This is the production choice;
    everything else in the pipeline is unchanged, and swapping backends only
    requires re-running ingest (the index records the backend that built it).

The ``Embedder`` protocol is intentionally tiny so a cloud embedding API can be
dropped in later without touching the retriever.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import Settings
from ..logging_setup import get_logger
from ..utils.text import normalize_text

logger = get_logger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Minimal contract every embedding backend must satisfy."""

    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32 matrix with L2-normalised rows."""
        ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise so dot product == cosine similarity.

    Zero rows (an embedding with no features) are left as zeros rather than
    producing NaNs, which would silently poison every similarity downstream.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Signed feature hashing embedder (offline, deterministic)."""

    #: Relative weight of each feature family.  Words carry most of the signal;
    #: bigrams capture short phrases ("carry over", "first response"); char
    #: 4-grams absorb typos and inflections.
    _WORD_WEIGHT = 1.0
    _BIGRAM_WEIGHT = 0.6
    _CHARGRAM_WEIGHT = 0.35

    def __init__(self, dim: int = 1024) -> None:
        if dim < 64:
            raise ValueError("dim must be >= 64")
        self.dim = dim
        self.name = f"hashing-{dim}"

    # -- internals ---------------------------------------------------------
    def _slot(self, token: str) -> tuple[int, float]:
        """Map a feature string to (index, sign).

        The sign comes from a second hash byte.  Signed hashing keeps collisions
        roughly zero-mean instead of always adding mass, which materially
        improves the quality of small hashed spaces.
        """
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, (1.0 if (value >> 63) & 1 else -1.0)

    def _features(self, text: str) -> list[tuple[str, float]]:
        cleaned = normalize_text(text).lower()
        words = [w for w in cleaned.split() if w.strip(".,;:!?()[]\"'")]

        features: list[tuple[str, float]] = []
        for word in words:
            features.append((f"w:{word}", self._WORD_WEIGHT))
        for left, right in zip(words, words[1:], strict=False):
            features.append((f"b:{left} {right}", self._BIGRAM_WEIGHT))
        # Character 4-grams over the whole string handle typos and morphology.
        compact = " ".join(words)
        for i in range(len(compact) - 3):
            features.append((f"c:{compact[i : i + 4]}", self._CHARGRAM_WEIGHT))
        return features

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text):
                index, sign = self._slot(feature)
                matrix[row, index] += sign * weight
        return _l2_normalise(matrix)


class SentenceTransformerEmbedder:
    """Semantic embeddings via ``sentence-transformers`` (optional)."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "EMBEDDING_BACKEND=sentence-transformers requires the optional "
                "package: pip install sentence-transformers"
            ) from exc

        logger.info("Loading sentence-transformers model %s", model_name)
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st-{model_name.split('/')[-1]}"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalise centrally below
            show_progress_bar=False,
        )
        return _l2_normalise(np.asarray(vectors, dtype=np.float32))


def build_embedder(settings: Settings) -> Embedder:
    """Factory returning the configured embedding backend."""
    backend = settings.embedding_backend.strip().lower()
    if backend in {"hashing", "hash", "local"}:
        return HashingEmbedder(dim=settings.embedding_dim)
    if backend in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(settings.embedding_model)
    raise ValueError(
        f"Unknown EMBEDDING_BACKEND={settings.embedding_backend!r}. "
        "Use 'hashing' or 'sentence-transformers'."
    )


def embedding_text(title: str, section: str, text: str) -> str:
    """Text used for scoring a chunk.

    Prepending the document title and section heading gives short chunks the
    topical context they otherwise lack - "180" means nothing, "travel expense -
    hotel caps: 180" does.  The citation still shows only the clean fields.
    """
    prefix = " - ".join(part for part in (title, section) if part.strip())
    return f"{prefix}. {text}" if prefix else text
