"""
src/core/settings.py

Single source of truth for all environment variables.

Every module that needs a config value imports from here — nothing reads
os.getenv() directly outside this file. This makes it trivial to audit
what config the app depends on and to swap values in tests.

Usage:
    from src.core.settings import settings
    conn = psycopg.connect(settings.pg_connection_string)
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file.

    Pydantic-settings reads each field from the matching env var name
    (case-insensitive). If the var is missing and no default is provided,
    startup fails with a clear validation error — no silent misconfiguration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # ignore unrecognised env vars silently
    )

    # ── PostgreSQL ────────────────────────────────────────────────────────
    pg_connection_string: str

    # ── OpenAI ────────────────────────────────────────────────────────────
    openai_api_key: str
    openai_chat_model: str = "gpt-5.4"
    openai_vision_model: str = "gpt-4o"
    openai_embeddings_model: str = "text-embedding-3-small"

    # ── LangSmith (optional — tracing only) ───────────────────────────────
    langsmith_tracing: bool = False
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str = ""
    langsmith_project: str = "capstone"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance.

    lru_cache ensures the .env file is read exactly once per process
    regardless of how many modules call get_settings().
    """
    return Settings()


# Module-level singleton — import this directly for convenience
settings = get_settings()