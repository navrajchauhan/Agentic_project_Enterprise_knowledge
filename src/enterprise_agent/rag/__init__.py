"""Retrieval-augmented generation building blocks.

``loaders``     read documents from the knowledge base
``chunking``    heading-aware, overlapping chunking
``embeddings``  offline feature hashing + optional sentence-transformers
``bm25``        lexical scoring (implemented from scratch)
``vector_store`` brute-force cosine store with persistence
``indexer``     build / save / load the hybrid index
``retriever``   RRF fusion, neighbour expansion, context rendering
"""

from __future__ import annotations

from .indexer import HybridIndex, Indexer
from .models import Chunk, Citation, Document, ScoredChunk
from .retriever import HybridRetriever

__all__ = [
    "Chunk",
    "Citation",
    "Document",
    "HybridIndex",
    "HybridRetriever",
    "Indexer",
    "ScoredChunk",
]
