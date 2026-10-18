"""Document loading from the knowledge base directory.

The enterprise knowledge base in this project is Markdown, but the loader is
deliberately format-agnostic: adding a new source type means adding one entry
to ``_LOADERS``.  PDF support is optional (``pip install pypdf``) so the core
project stays dependency-light.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from ..logging_setup import get_logger
from .models import Document

logger = get_logger(__name__)

#: Supported extensions.  Order matters only for readability of the error text.
TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst"}

_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def _read_text_file(path: Path) -> str:
    """Read a text document, tolerating the BOM/latin-1 edge cases we see in
    exported corporate documents."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("Could not decode %s as UTF-8; falling back to latin-1", path.name)
        return path.read_text(encoding="latin-1")


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF when ``pypdf`` is installed."""
    try:
        from pypdf import PdfReader  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PDF support needs the optional 'pypdf' package: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages).strip()


#: Extension -> reader.  Adding a format is one line.
_LOADERS: dict[str, Callable[[Path], str]] = {
    **{suffix: _read_text_file for suffix in TEXT_SUFFIXES},
    ".pdf": _read_pdf,
}


def _derive_title(text: str, fallback: str) -> str:
    """Use the first ``# heading`` as the title, else prettify the file name."""
    match = _H1_RE.search(text)
    if match:
        return match.group(1).strip()
    return fallback.replace("_", " ").replace("-", " ").title()


def load_documents(knowledge_base_dir: Path) -> list[Document]:
    """Load every supported document from a directory (non-recursive by design).

    Returns documents sorted by file name so that chunk ids and therefore
    evaluation numbers are stable between runs.

    Raises
    ------
    FileNotFoundError
        When the directory does not exist, with a hint about ``INGEST``.
    ValueError
        When the directory exists but contains no usable document.
    """
    if not knowledge_base_dir.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {knowledge_base_dir}. "
            "Create it and add .md/.txt documents (or fix KNOWLEDGE_BASE_DIR in .env)."
        )

    documents: list[Document] = []
    for path in sorted(knowledge_base_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        reader = _LOADERS.get(suffix)
        if reader is None:
            logger.debug("Skipping unsupported file type: %s", path.name)
            continue

        try:
            text = reader(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill ingest
            logger.error("Failed to read %s: %s", path.name, exc)
            continue

        if not text.strip():
            logger.warning("Skipping empty document: %s", path.name)
            continue

        doc_id = path.stem
        documents.append(
            Document(
                doc_id=doc_id,
                title=_derive_title(text, doc_id),
                source_path=path.name,
                text=text,
                metadata={"suffix": suffix, "characters": len(text)},
            )
        )

    if not documents:
        raise ValueError(
            f"No readable documents found in {knowledge_base_dir}. "
            f"Supported extensions: {', '.join(sorted(_LOADERS))}"
        )

    logger.info("Loaded %d document(s) from %s", len(documents), knowledge_base_dir.name)
    return documents
