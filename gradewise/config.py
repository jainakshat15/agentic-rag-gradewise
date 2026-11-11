"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 1 · Day 1: Your settings, in one place                      │
│                                                                      │
│  What this file does                                                 │
│    Reads your `.env` file once and exposes every setting as a typed  │
│    attribute: `settings.chat_model`, `settings.budget_mode`, ...     │
│    Model names live ONLY here (and in .env), never hard-coded in a   │
│    notebook, so swapping providers is a one-line change.             │
│                                                                      │
│  Use it                                                              │
│    from gradewise.config import settings                             │
│    print(settings.chat_model)                                        │
│                                                                      │
│  ⏭ Used by: models.py, every notebook's setup cell                  │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# 💡 The repo root is the folder that holds pyproject.toml. We find it from this
#    file's location, so paths work no matter which week folder you run from.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"  # git-ignored: vector stores, databases, reports


class Settings(BaseSettings):
    """Every knob in the course. Values come from `.env`; defaults are sensible."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # 💡 unrelated keys in .env are fine
    )

    # ── 🔑 API keys (only the provider you use needs to be set) ──────────
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    langsmith_api_key: SecretStr | None = None

    # ── 🧠 Models, written as "provider:model" for init_chat_model ───────
    # 🚀 PRO: the course uses three "tiers" so you can trade cost for quality.
    chat_model: str = Field(default="openai:gpt-5.4-mini", description="Everyday model")
    smart_model: str = Field(default="openai:gpt-5.5", description="Grading model")
    vision_model: str = Field(default="openai:gpt-5.5", description="Handwriting model")
    embedding_model: str = Field(default="openai:text-embedding-3-small")

    # ── 💰 Budget mode: when true, every tier uses chat_model ────────────
    budget_mode: bool = False

    # ── 🔭 Observability ─────────────────────────────────────────────────
    langsmith_tracing: bool = False
    langsmith_project: str = "gradewise"

    # ── 🏠 Local models (Ollama) ─────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"

    @property
    def grading_model(self) -> str:
        """The model used for grading: falls back to chat_model in budget mode."""
        return self.chat_model if self.budget_mode else self.smart_model

    @property
    def transcription_model(self) -> str:
        """The vision model: falls back to chat_model in budget mode (must support images!)."""
        return self.chat_model if self.budget_mode else self.vision_model

    @property
    def provider(self) -> str:
        """The provider part of chat_model, e.g. 'openai'."""
        return self.chat_model.split(":", 1)[0] if ":" in self.chat_model else "openai"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once. Call `get_settings.cache_clear()` after editing .env."""
    return Settings()


settings = get_settings()
