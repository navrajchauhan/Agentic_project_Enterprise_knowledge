"""A small, persisted dense vector store.

For an enterprise knowledge base of thousands (not billions) of chunks, a
brute-force cosine scan over a NumPy matrix is the right engineering call: it
is ~1 ms for 10k chunks, has zero operational overhead, and keeps the project
runnable with no external service.  A production deployment would swap this
class for pgvector / Qdrant / FAISS behind the same three methods
(``add`` / ``search`` / ``save``).

The index is persisted as two files:

``vectors.npy``      float32 matrix, rows aligned with ``vector_ids.json``
``vector_ids.json``  chunk ids in row order
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..logging_setup import get_logger

logger = get_logger(__name__)


class VectorStore:
    """In-memory cosine-similarity store with save/load support."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)

    # -- population --------------------------------------------------------
    def add(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        """Append chunk ids and their (row-normalised) vectors."""
        if len(chunk_ids) != len(vectors):
            raise ValueError(
                f"Got {len(chunk_ids)} ids but {len(vectors)} vectors; they must align"
            )
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors with shape (n, {self.dim}), got {vectors.shape}")
        if len(chunk_ids) == 0:
            return

        self._ids.extend(chunk_ids)
        self._matrix = (
            np.vstack([self._matrix, vectors.astype(np.float32)])
            if self._matrix.size
            else vectors.astype(np.float32)
        )

    # -- querying ----------------------------------------------------------
    def search(self, query_vector: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
        """Return ``(chunk_id, cosine_similarity)`` pairs, best first."""
        if self._matrix.size == 0:
            return []
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Both sides are L2-normalised by the embedder, so a dot product is the
        # cosine similarity.  Flatten to keep the result a simple list.
        similarities = (self._matrix @ query_vector.reshape(-1)).astype(np.float32)
        k = min(top_k, len(similarities))
        # ``argpartition`` is O(n) instead of O(n log n); we only sort the top k.
        top_indices = np.argpartition(-similarities, k - 1)[:k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        return [
            (self._ids[int(index)], float(similarities[int(index)]))
            for index in top_indices
            if similarities[int(index)] > 0.0
        ]

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    # -- persistence -------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._matrix)
        (directory / "vector_ids.json").write_text(
            json.dumps(self._ids, indent=2), encoding="utf-8"
        )
        logger.debug("Saved %d vectors to %s", len(self._ids), directory)

    @classmethod
    def load(cls, directory: Path, dim: int) -> VectorStore:
        matrix_path = directory / "vectors.npy"
        ids_path = directory / "vector_ids.json"
        if not matrix_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"No persisted vector store in {directory}")

        matrix = np.load(matrix_path).astype(np.float32)
        ids = json.loads(ids_path.read_text(encoding="utf-8"))

        if matrix.ndim != 2 or matrix.shape[1] != dim:
            raise ValueError(
                f"Persisted vectors have shape {matrix.shape} but the current embedder "
                f"produces {dim}-d vectors. Re-run ingest (the embedding backend changed)."
            )
        if len(ids) != matrix.shape[0]:
            raise ValueError("vector_ids.json does not match vectors.npy; re-run ingest")

        store = cls(dim)
        store._ids = list(ids)
        store._matrix = matrix
        return store
