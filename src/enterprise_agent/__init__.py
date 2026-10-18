"""Enterprise RAG Agent.

A retrieval-augmented, tool-using agent for enterprise documents:

* hybrid retrieval (BM25 + dense vectors fused with Reciprocal Rank Fusion)
* source-grounded answers with inline citations and an explicit "no evidence"
  path instead of guessing
* stateful LangGraph workflow with per-thread conversation memory
* six enterprise actions exposed both in-process and over MCP
* input/output guardrails (prompt injection, PII, citation + grounding checks)
* an offline evaluation harness (retrieval metrics + LLM-as-a-judge)

Package layout::

    config.py         settings loaded from .env (pydantic-settings)
    utils/            tokenisation, JSON parsing, small shared helpers
    rag/              loaders, chunker, embeddings, BM25, vector store, retriever
    tools/            enterprise actions + tool registry (single source of truth)
    mcp/              MCP stdio server and client adapters
    guardrails/       input and output safety checks
    agent/            LangGraph state, nodes, graph, runner
    evaluation/       retrieval metrics, LLM judge, evaluation runner
    cli.py            command line interface
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.0.0"
