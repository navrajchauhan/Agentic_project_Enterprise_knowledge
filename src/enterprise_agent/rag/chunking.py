"""Heading-aware chunking.

Why not naive fixed-size chunking?
    Enterprise policies are organised as sections ("Hotel Caps", "Password
    Rules").  A chunk that starts in the middle of one rule and ends in another
    is poison for retrieval: the embedding mixes two topics and the citation
    points at the wrong place.

The strategy is a two-step split:

1. **Split on Markdown headings** so every chunk belongs to exactly one
   section.  The heading path is stored on the chunk and is the citation
   label the user sees.
2. **Slide a word window inside each section**, with overlap, so long sections
   are still digestible by the LLM and a sentence straddling a boundary is
   retrievable from both neighbours.

``chunk_index`` is document-global and contiguous, which is what makes
neighbour expansion (pulling in ``chunk_index - 1`` / ``+ 1``) work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..logging_setup import get_logger
from .models import Chunk, Document

logger = get_logger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
#: Fenced blocks are treated as opaque text (they may contain '#').
_FENCE_RE = re.compile(r"^\s*```")


@dataclass(frozen=True)
class Section:
    """A contiguous run of body text belonging to one heading path."""

    heading_path: str
    text: str


def split_sections(text: str) -> list[Section]:
    """Split Markdown into ``Section`` objects (heading path + body).

    The H1 document title is dropped from the heading path because it is
    already stored on the document and repeating it in every chunk would bias
    the lexical scores.
    """
    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title)
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            path = " - ".join(title for _, title in heading_stack)
            sections.append(Section(heading_path=path, text=body))

    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue

        match = None if in_fence else _HEADING_RE.match(line)
        if match is None:
            buffer.append(line)
            continue

        # New heading: close the previous section, then update the heading stack.
        flush()
        buffer = []
        level = len(match.group(1))
        title = match.group(2).strip()

        if level == 1:
            # Document title: not part of the citation path.
            heading_stack = []
        else:
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

    flush()
    return sections


def split_words(words: list[str], size: int, overlap: int) -> list[list[str]]:
    """Slide a ``size``-word window with ``overlap`` words of context.

    A trailing window shorter than ``overlap`` is merged into its predecessor
    instead of becoming a stubby chunk (stub chunks retrieve badly: they have
    too few terms for BM25 and too little meaning for the dense model).
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    if len(words) <= size:
        return [list(words)]

    windows: list[list[str]] = []
    step = size - overlap
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        windows.append(list(words[start:end]))
        if end >= len(words):
            break
        start += step

    if len(windows) > 1 and len(windows[-1]) <= overlap:
        # The slices are contiguous, so extending the previous window with the
        # tail reconstructs the original text exactly.
        windows[-2].extend(windows[-1])
        windows.pop()

    return windows


def chunk_document(
    document: Document,
    *,
    size_words: int = 120,
    overlap_words: int = 25,
    min_words: int = 8,
) -> list[Chunk]:
    """Turn one document into a list of cited, ordered chunks."""
    chunks: list[Chunk] = []
    index = 0

    for section in split_sections(document.text):
        words = section.text.split()
        if len(words) < min_words:
            # Title-only or whitespace-only sections carry no retrievable fact.
            logger.debug(
                "Skipping tiny section %r in %s (%d words)",
                section.heading_path,
                document.doc_id,
                len(words),
            )
            continue

        for window_index, window in enumerate(split_words(words, size_words, overlap_words)):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::{index:04d}",
                    doc_id=document.doc_id,
                    title=document.title,
                    section=section.heading_path or "Overview",
                    text=" ".join(window),
                    chunk_index=index,
                    word_start=window_index * (size_words - overlap_words),
                    metadata={
                        "source_path": document.source_path,
                        "section_words": len(words),
                    },
                )
            )
            index += 1

    logger.debug("Chunked %s into %d chunk(s)", document.doc_id, len(chunks))
    return chunks


def chunk_documents(
    documents: list[Document],
    *,
    size_words: int = 120,
    overlap_words: int = 25,
    min_words: int = 8,
) -> list[Chunk]:
    """Chunk every document, preserving document order."""
    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(
            chunk_document(
                document,
                size_words=size_words,
                overlap_words=overlap_words,
                min_words=min_words,
            )
        )
    if not all_chunks:
        raise ValueError("Chunking produced no chunks; are the documents empty?")
    return all_chunks
