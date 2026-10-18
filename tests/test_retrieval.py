"""BM25 scoring behaviour and hybrid fusion."""

from __future__ import annotations

import pytest

from enterprise_agent.rag.bm25 import BM25Index
from enterprise_agent.rag.embeddings import HashingEmbedder, embedding_text
from enterprise_agent.rag.vector_store import VectorStore
from enterprise_agent.utils.text import tokenize


CORPUS = [
    "annual leave is 24 days per year and 5 days may be carried over",
    "passwords must be 14 characters and rotated every 90 days",
    "the hotel cap is 180 usd domestic and 250 usd international",
    "p1 incidents get a first response within 15 minutes",
]


@pytest.fixture()
def bm25() -> BM25Index:
    return BM25Index([tokenize(doc) for doc in CORPUS])


def test_rate_terms_rank_the_right_document(bm25: BM25Index) -> None:
    top = bm25.search("hotel cap international")
    assert top, "expected at least one match"
    assert CORPUS[top[0][0]].startswith("the hotel cap")


def test_unseen_terms_score_zero(bm25: BM25Index) -> None:
    assert bm25.search("quantum blockchain") == []


def test_empty_query_returns_nothing(bm25: BM25Index) -> None:
    assert bm25.search("") == []
    assert bm25.search("   ") == []


def test_idf_is_never_negative(bm25: BM25Index) -> None:
    # A term present in every document would be negative with the classic
    # formula; the smoothed variant keeps it >= 0. Unseen terms are 0.
    assert all(bm25.idf(token) > 0 for token in {"day", "usd"})
    assert bm25.idf("nonexistentterm") == 0.0


def test_explain_breaks_down_contributions(bm25: BM25Index) -> None:
    contributions = bm25.explain("hotel cap", doc_index=2)
    terms = [term for term, _score in contributions]
    assert "hotel" in terms and "cap" in terms
    assert all(score > 0 for _term, score in contributions)


def test_tokenizer_normalises_plurals_and_stopwords() -> None:
    assert tokenize("The days of the Passwords") == ["day", "password"]


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(dim=256)
    first = embedder.encode(["hotel cap for travel"])
    second = embedder.encode(["hotel cap for travel"])

    assert first.shape == (1, 256)
    assert (first == second).all()
    norm = float((first[0] ** 2).sum() ** 0.5)
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_hashing_embedder_similar_texts_are_closer() -> None:
    embedder = HashingEmbedder(dim=512)
    vectors = embedder.encode(
        [
            "hotel cap for domestic travel",
            "hotel caps for domestic travel",  # plural typo
            "password rotation policy",
        ]
    )
    similar = float(vectors[0] @ vectors[1])
    different = float(vectors[0] @ vectors[2])
    assert similar > different


def test_vector_store_roundtrip(tmp_dir) -> None:
    embedder = HashingEmbedder(dim=128)
    texts = ["alpha beta", "gamma delta", "epsilon zeta"]
    matrix = embedder.encode(texts)

    store = VectorStore(dim=128)
    store.add([f"c{index}" for index in range(3)], matrix)
    store.save(tmp_dir)

    loaded = VectorStore.load(tmp_dir, dim=128)
    assert len(loaded) == 3

    hits = loaded.search(embedder.encode(["gamma delta"])[0], top_k=2)
    assert hits[0][0] == "c1"
    assert hits[0][1] > 0.5


def test_hybrid_retriever_returns_ranked_unique_hits(retriever) -> None:
    hits = retriever.search("hotel cap for international travel", top_k=4)

    assert hits, "expected hits from the knowledge base"
    assert len(hits) <= 4
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert len({hit.chunk_id for hit in hits}) == len(hits)
    # At least one hit must come from the right document.
    assert any(hit.chunk.doc_id == "travel_expense_policy" for hit in hits)
    # The fused score must be explainable by the underlying ranks.
    assert all(hit.bm25_rank or hit.dense_rank for hit in hits)


def test_retriever_document_filter(retriever) -> None:
    hits = retriever.search("password rotation", top_k=4, doc_ids=["it_security_policy"])
    assert hits
    assert {hit.chunk.doc_id for hit in hits} == {"it_security_policy"}


def test_retriever_handles_empty_query(retriever) -> None:
    assert retriever.search("") == []
    assert retriever.search("    ") == []


def test_neighbour_expansion_adds_context(retriever) -> None:
    hits = retriever.search("password rotation interval", top_k=3)
    expanded = [hit for hit in hits if hit.expanded_text]
    assert expanded, "neighbour expansion should populate expanded_text"
    assert any(len(hit.context_text) >= len(hit.chunk.text) for hit in hits)


def test_context_block_is_numbered_and_citable(retriever) -> None:
    hits = retriever.search("annual leave carry over", top_k=3)
    block = retriever.context_block(hits, max_chars=2000)

    assert block.startswith("[1] source:")
    assert "chunk_id=" in block
    # Every hit that fits the budget gets its own numbered block.
    assert block.count("[1] source:") == 1
    assert "[2] source:" in block or len(hits) == 1


def test_embedding_text_includes_topic_prefix() -> None:
    text = embedding_text("Travel Policy", "Accommodation", "The cap is 180 USD.")
    assert text.startswith("Travel Policy - Accommodation.")
