"""Groq chat model construction.

Groq is used through ``langchain-groq`` so the model integrates with LangGraph's
message conventions and tool calling with no custom glue.

Operational choices:

* ``temperature=0`` by default - policy answers should be reproducible.
* ``max_retries=2`` - Groq occasionally returns 429s on the free tier; the
  retries are handled inside the client so nodes stay simple.
* an explicit timeout, because a hung request must not hang the graph: the
  runner also enforces a per-question wall-clock budget.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from ..config import Settings
from ..logging_setup import get_logger

logger = get_logger(__name__)


def build_groq_model(settings: Settings, *, model_name: str) -> BaseChatModel:
    """Create a ``ChatGroq`` client for ``model_name``.

    Raises a clear, actionable error when the key or the SDK is missing instead
    of failing later with an opaque 401.
    """
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is empty. Either add your key to .env or run in offline mode "
            "(offline mode is automatic when the key is absent)."
        )

    try:
        from langchain_groq import ChatGroq  # noqa: PLC0415 - optional at import time
    except ImportError as exc:
        raise RuntimeError(
            "langchain-groq is not installed. Run: pip install -r requirements.txt"
        ) from exc

    logger.debug("Building Groq model %s", model_name)
    return ChatGroq(
        model=model_name,
        groq_api_key=settings.groq_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=2,
    )
