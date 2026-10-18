"""Application configuration.

Every tunable lives here and is read from environment variables / the ``.env``
file, which keeps configuration out of the code and makes the agent easy to
tune per deployment (an SDE-1 habit worth keeping).

Design notes
------------
* ``pydantic-settings`` validates values at startup, so a typo such as
  ``TOP_K=abc`` fails immediately with a clear message instead of blowing up
  halfway through a request.
* Relative paths are resolved against the *project root* (the directory that
  contains ``src/``), not the current working directory.  That means the CLI
  works the same whether you run it from the repo root or from anywhere else.
* ``Settings.is_offline`` is the single switch the rest of the codebase checks
  to decide between the live Groq model and the deterministic offline model.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ``config.py`` is at <root>/src/enterprise_agent/config.py, so the project root
# is three levels up.  Using this instead of ``Path.cwd()`` keeps paths stable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: Path) -> Path:
    """Make a configured path absolute, relative to the project root."""
    return path if path.is_absolute() else (PROJECT_ROOT / path)


class Settings(BaseSettings):
    """Typed, validated view of the application configuration."""

    model_config = SettingsConfigDict(
        # ``.env`` next to the project root wins; a second entry lets a caller
        # drop a local override file next to the process for containers/tests.
        env_file=(PROJECT_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unrelated variables (CI, shells)
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ Groq
    groq_api_key: str | None = Field(
        default=None,
        description="Groq API key. Empty means offline mode.",
    )
    llm_model: str = "llama-3.3-70b-versatile"
    llm_fast_model: str = "llama-3.1-8b-instant"
    llm_judge_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1200
    llm_timeout_seconds: float = 60.0

    # ------------------------------------------------------------- Ingestion
    knowledge_base_dir: Path = Path("knowledge_base")
    index_dir: Path = Path("data/index")
    chunk_size_words: int = Field(default=120, ge=20, le=1000)
    chunk_overlap_words: int = Field(default=25, ge=0, le=500)

    # ------------------------------------------------------------- Retrieval
    top_k: int = Field(default=6, ge=1, le=50)
    bm25_weight: float = Field(default=0.5, ge=0.0)
    dense_weight: float = Field(default=0.5, ge=0.0)
    rrf_k: int = Field(default=60, ge=1)
    enable_neighbor_expansion: bool = True
    max_context_chars: int = Field(default=6000, ge=500)

    # ------------------------------------------------------------ Embeddings
    embedding_backend: str = "hashing"
    embedding_dim: int = Field(default=1024, ge=64)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ----------------------------------------------------------------- Agent
    max_tool_steps: int = Field(default=4, ge=1, le=12)
    max_query_rewrites: int = Field(default=1, ge=0, le=3)
    min_grounding_score: float = Field(default=0.55, ge=0.0, le=1.0)
    checkpoint_backend: str = "memory"
    checkpoint_db: Path = Path("data/checkpoints.sqlite")

    # ------------------------------------------------------------ Tools / MCP
    tool_provider: str = "auto"
    mcp_server_command: str = ".venv\\Scripts\\python.exe"
    mcp_server_module: str = "enterprise_agent.mcp.server"
    mcp_call_timeout_seconds: float = Field(default=30.0, gt=0)
    action_db: Path = Path("data/actions.sqlite")

    # ------------------------------------------------------------- Guardrails
    max_input_chars: int = Field(default=2000, ge=50)
    redact_pii: bool = True
    block_prompt_injection: bool = True
    require_citations: bool = True
    #: Emails on this domain are treated as internal contact details and are NOT
    #: redacted from answers (they are often the answer itself).
    internal_email_domain: str = "company.example"

    # ------------------------------------------------------------- Evaluation
    eval_dataset: Path = Path("evals/golden_set.json")
    eval_top_k: int = Field(default=5, ge=1, le=20)
    eval_report_dir: Path = Path("reports")
    enable_llm_judge: bool = True

    # ---------------------------------------------------------------- Runtime
    log_level: str = "INFO"

    # ------------------------------------------------------------- Validators
    @field_validator("groq_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        """Treat ``GROQ_API_KEY=`` (empty in .env) as 'not configured'."""
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        """Reject combinations that would silently produce bad behaviour."""
        if self.bm25_weight + self.dense_weight <= 0:
            raise ValueError("BM25_WEIGHT + DENSE_WEIGHT must be greater than 0")
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("CHUNK_OVERLAP_WORDS must be smaller than CHUNK_SIZE_WORDS")
        if self.tool_provider not in {"auto", "mcp", "direct"}:
            raise ValueError("TOOL_PROVIDER must be one of: auto, mcp, direct")
        if self.checkpoint_backend not in {"memory", "sqlite"}:
            raise ValueError("CHECKPOINT_BACKEND must be one of: memory, sqlite")
        return self

    # ------------------------------------------------------------- Convenience
    @property
    def is_offline(self) -> bool:
        """True when no Groq key is configured.

        The graph, retrieval, guardrails and tools still work in this mode; only
        the text generation is replaced by a deterministic extractive model so
        the project is demoable and testable without paid API calls.
        """
        return not self.groq_api_key

    @property
    def knowledge_base_path(self) -> Path:
        return _resolve(self.knowledge_base_dir)

    @property
    def index_path(self) -> Path:
        return _resolve(self.index_dir)

    @property
    def checkpoint_path(self) -> Path:
        return _resolve(self.checkpoint_db)

    @property
    def action_db_path(self) -> Path:
        return _resolve(self.action_db)

    @property
    def eval_dataset_path(self) -> Path:
        return _resolve(self.eval_dataset)

    @property
    def report_path(self) -> Path:
        return _resolve(self.eval_report_dir)

    def ensure_directories(self) -> None:
        """Create the writable directories the application expects."""
        for path in (self.index_path, self.report_path, self.action_db_path.parent):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process.

    ``lru_cache`` keeps the object identity stable so callers can rely on
    ``get_settings() is get_settings()`` - handy for tests and for passing the
    settings around without a dependency-injection framework.
    """
    return Settings()
