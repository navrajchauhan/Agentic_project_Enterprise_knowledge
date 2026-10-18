"""Chunking: heading structure, window sizes, overlap and stable ids."""

from __future__ import annotations

from enterprise_agent.rag.chunking import (
    chunk_document,
    chunk_documents,
    split_sections,
    split_words,
)
from enterprise_agent.rag.loaders import load_documents
from enterprise_agent.rag.models import Document


def test_split_sections_tracks_heading_paths() -> None:
    text = """# Doc Title

Intro paragraph.

## First Section

Body one.

### Nested

Body two.

## Second Section

Body three.
"""
    sections = split_sections(text)
    paths = [section.heading_path for section in sections]
    assert paths == ["", "First Section", "First Section - Nested", "Second Section"]
    assert "Body two." in sections[2].text
    # The H1 title must not leak into the citation path.
    assert all("Doc Title" not in path for path in paths)


def test_split_words_respects_size_and_overlap() -> None:
    words = [f"w{index}" for index in range(25)]
    windows = split_words(words, size=10, overlap=3)

    assert all(len(window) <= 11 for window in windows)  # tail merge may add up to overlap
    # Overlap is real: the last 3 words of window 1 start window 2.
    assert windows[0][-3:] == windows[1][:3]
    # Nothing is lost: concatenating unique-in-order recovers the input.
    reconstructed: list[str] = []
    for window in windows:
        for word in window:
            if not reconstructed or word not in reconstructed[-3:]:
                reconstructed.append(word)
    assert reconstructed == words


def test_split_words_merges_short_tail() -> None:
    words = [f"w{index}" for index in range(21)]
    windows = split_words(words, size=10, overlap=3)
    assert len(windows[-1]) > 3  # tail folded into the previous window


def test_chunk_document_metadata_and_ids() -> None:
    document = Document(
        doc_id="sample",
        title="Sample Policy",
        source_path="sample.md",
        text="# Sample Policy\n\n## Alpha\n\n" + " ".join(f"a{i}" for i in range(200))
        + "\n\n## Beta\n\n" + " ".join(f"b{i}" for i in range(150)),
    )
    chunks = chunk_document(document, size_words=50, overlap_words=10)

    assert len(chunks) >= 4
    # Chunk ids are unique, sequential and prefixed with the document id.
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(chunk.chunk_id.startswith("sample::") for chunk in chunks)
    # Section labels are usable as citations.
    assert {chunk.section for chunk in chunks} == {"Alpha", "Beta"}
    # Word windows stay near the configured size.
    assert all(len(chunk.text.split()) <= 60 for chunk in chunks)


def test_chunk_documents_keeps_document_order(project_root) -> None:
    documents = load_documents(project_root / "knowledge_base")
    chunks = chunk_documents(documents, size_words=120, overlap_words=25)

    doc_order = [chunk.doc_id for chunk in chunks]
    # Chunks must be grouped per document, so a document's chunks are contiguous.
    # That is what makes neighbour expansion by chunk_index valid.
    runs = [doc_id for index, doc_id in enumerate(doc_order) if index == 0 or doc_order[index - 1] != doc_id]
    assert len(runs) == len(set(runs)) == 5
    assert len({chunk.doc_id for chunk in chunks}) == 5
    assert len(chunks) > 25


def test_tiny_sections_are_dropped() -> None:
    document = Document(
        doc_id="tiny",
        title="Tiny",
        source_path="tiny.md",
        text="# Tiny\n\n## Empty\n\nn/a\n",
    )
    assert chunk_document(document) == []
