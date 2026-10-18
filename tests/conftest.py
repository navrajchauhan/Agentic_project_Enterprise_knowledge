"""Shared pytest fixtures.

Everything here runs **offline**:

* ``GROQ_API_KEY`` is forced empty, so ``build_chat_model`` returns the
  deterministic extractive model instead of calling Groq;
* the index, SQLite database and reports live in a temporary directory, so the
  test suite never touches the developer's real ``data/`` folder;
* the index is built once per session (it is the expensive fixture) and reused.

That combination is what makes the agent testable in CI: no network, no keys,
no fixtures to maintain by hand.
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from enterprise_agent.config import Settings  # noqa: E402 - after sys.path setup


@pytest.fixture(scope="session")
def workdir() -> Path:
    """Scratch directory for generated artefacts.

    Deliberately *not* pytest's ``tmp_path``: on Windows that creates a 0700
    temporary directory, which the confined test process cannot then re-open.
    A plain directory under the (ignored) ``.tmp/`` tree avoids that.
    """
    base = PROJECT_ROOT / ".tmp" / "pytest" / uuid.uuid4().hex[:8]
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture()
def tmp_dir(workdir: Path, request: pytest.FixtureRequest) -> Path:
    """Per-test subdirectory of ``workdir``."""
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in request.node.name)[:50]
    path = workdir / safe_name
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def settings(workdir: Path) -> Settings:
    """Isolated, offline settings shared by the whole session."""
    base = workdir / "agent-data"
    base.mkdir(parents=True, exist_ok=True)
    return Settings(
        groq_api_key=None,  # force offline mode
        knowledge_base_dir=PROJECT_ROOT / "knowledge_base",
        index_dir=base / "index",
        chunk_size_words=120,
        chunk_overlap_words=25,
        top_k=6,
        bm25_weight=0.5,
        dense_weight=0.5,
        enable_neighbor_expansion=True,
        max_context_chars=6000,
        embedding_backend="hashing",
        embedding_dim=512,
        max_tool_steps=4,
        max_query_rewrites=1,
        min_grounding_score=0.55,
        checkpoint_backend="memory",
        checkpoint_db=base / "checkpoints.sqlite",
        tool_provider="direct",  # no subprocess in unit tests
        action_db=base / "actions.sqlite",
        max_input_chars=2000,
        redact_pii=True,
        block_prompt_injection=True,
        require_citations=True,
        eval_dataset=PROJECT_ROOT / "evals" / "golden_set.json",
        eval_top_k=5,
        eval_report_dir=base / "reports",
        enable_llm_judge=True,
        log_level="WARNING",
    )


@pytest.fixture(scope="session")
def index(settings: Settings):
    """Hybrid index built once from the bundled knowledge base."""
    from enterprise_agent.rag.indexer import Indexer

    return Indexer(settings).build(force=True)


@pytest.fixture(scope="session")
def retriever(settings: Settings, index):
    from enterprise_agent.rag.retriever import HybridRetriever

    return HybridRetriever(index, settings)


@pytest.fixture(scope="session")
def store(settings: Settings):
    from enterprise_agent.tools.store import ActionStore

    return ActionStore(settings.action_db_path)


@pytest.fixture(scope="session")
def registry(settings: Settings, store, retriever):
    from enterprise_agent.tools.registry import build_registry

    return build_registry(store=store, retriever=retriever, settings=settings)


@pytest.fixture(scope="session")
def runner(settings: Settings):
    """Fully wired agent, offline."""
    from enterprise_agent.agent.runner import AgentRunner

    agent = AgentRunner(settings)
    yield agent
    agent.close()
