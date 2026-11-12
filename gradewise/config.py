"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 1 · Day 1: Your settings, in one place                      │
│                                                                      │
│  What this file does                                                 │
│    Reads your `.env` file once and exposes every setting as a typed  │
│    attribute: `settings.model_mode`, `settings.chat_model`, ...      │
│    Model names live ONLY here (and in .env), never hard-coded in a   │
│    notebook, so switching is a one-line change.                      │
│                                                                      │
│  💰 MODEL_MODE picks every model at once (one OPENROUTER_API_KEY):   │
│      cheap  (default) → openrouter:openai/gpt-6-luna                 │
│      free             → openrouter:openrouter/free ($0, rate-limited)│
│      custom           → whatever CHAT_MODEL / SMART_MODEL /          │
│                         VISION_MODEL / EMBEDDING_MODEL say           │
│                                                                      │
│  Use it                                                              │
│    from gradewise.config import settings                             │
│    print(settings.model_mode, settings.chat_model)                   │
│                                                                      │
│  ⏭ Used by: models.py, datasets.py, every notebook's setup cell     │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 💡 The repo root is the folder that holds pyproject.toml. We find it from this
#    file's location, so paths work no matter which week folder you run from.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"  # git-ignored: vector stores, databases, reports

ModelMode = Literal["cheap", "free", "custom"]
DataSize = Literal["full", "small"]
MODEL_MODES: tuple[str, ...] = ("cheap", "free", "custom")
DATA_SIZES: tuple[str, ...] = ("full", "small")

# 🧠 The models each preset mode uses. `cheap` and `free` IGNORE the *_MODEL variables in .env,
#    so switching is one line. 💡 In `cheap`, all three tiers are the same model: the SMART tier
#    (grading) just thinks harder (a higher reasoning effort, see gradewise/models.py).
#    Both presets go through OpenRouter (https://openrouter.ai), so ONE key (OPENROUTER_API_KEY) does it all.
#    💡 openrouter/free is OpenRouter's free-models ROUTER: for each request it keeps only the free models
#       that support what the request needs (images, tools, structured output) and picks one at random.
MODE_MODELS: dict[str, dict[str, str]] = {
    "cheap": {
        "chat": "openrouter:openai/gpt-6-luna",
        "smart": "openrouter:openai/gpt-6-luna",
        "vision": "openrouter:openai/gpt-6-luna",
        "embedding": "openrouter:openai/text-embedding-3-small",
    },
    "free": {
        "chat": "openrouter:openrouter/free",
        "smart": "openrouter:openrouter/free",
        "vision": "openrouter:openrouter/free",
        "embedding": "openrouter:nvidia/nemotron-3.5-embed-1b:free",
    },
}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
FREE_ROUTER = "openrouter/free"

# 🔑 Which environment variable each provider needs. Ollama runs locally: no key.
PROVIDER_KEYS: dict[str, str | None] = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "ollama": None,
}

# ⏳ Free tiers have per-minute rate limits, so `free` mode runs at most this many calls at once.
FREE_MAX_CONCURRENCY = 2


def provider_of(model: str) -> str:
    """'openai:gpt-6-luna' → 'openai'."""
    return model.split(":", 1)[0] if ":" in model else "openai"


class Settings(BaseSettings):
    """Every knob in the course. Values come from `.env`; defaults are sensible."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # 💡 unrelated keys in .env are fine (an old BUDGET_MODE=... is simply ignored)
        populate_by_name=True,
    )

    # ── 💰 The two switches ──────────────────────────────────────────────
    model_mode: ModelMode = Field(default="cheap", description="cheap | free | custom")
    data_size: DataSize = Field(default="full", description="full | small (fewer PDFs, questions, students)")

    # ── 🔑 API keys: cheap and free need only OPENROUTER_API_KEY ────────
    openrouter_api_key: SecretStr | None = None
    # 💡 custom mode only, for models you call directly at a provider:
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    langsmith_api_key: SecretStr | None = None

    # ── 🧠 Custom models, "provider:model" for init_chat_model ───────────
    # ⚠️ Only used when MODEL_MODE=custom. Read the RESOLVED models from the properties below
    #    (settings.chat_model, settings.grading_model, ...), never from these fields.
    custom_chat_model: str = Field(
        default="openrouter:openai/gpt-6-luna", validation_alias=AliasChoices("CHAT_MODEL", "chat_model")
    )
    custom_smart_model: str = Field(
        default="openrouter:openai/gpt-6-luna", validation_alias=AliasChoices("SMART_MODEL", "smart_model")
    )
    custom_vision_model: str = Field(
        default="openrouter:openai/gpt-6-luna", validation_alias=AliasChoices("VISION_MODEL", "vision_model")
    )
    custom_embedding_model: str = Field(
        default="openrouter:openai/text-embedding-3-small",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "embedding_model"),
    )

    # ── 🔭 Observability ─────────────────────────────────────────────────
    langsmith_tracing: bool = False
    langsmith_project: str = "gradewise"

    # ── 🏠 Local models (Ollama) ─────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"

    @field_validator("model_mode", "data_size", mode="before")
    @classmethod
    def _normalise_choice(cls, value: object, info: ValidationInfo) -> str:
        """`MODEL_MODE=Free ` → "free"; an empty value → the default."""
        defaults = {"model_mode": "cheap", "data_size": "full"}
        text = str(value or "").strip().strip("\"'").lower()
        return text or defaults[info.field_name]

    @field_validator(
        "custom_chat_model", "custom_smart_model", "custom_vision_model", "custom_embedding_model", mode="before"
    )
    @classmethod
    def _strip_model(cls, value: object) -> object:
        return value.strip().strip("\"'") if isinstance(value, str) else value

    # ── 🧠 The resolved models (what GradeWise actually uses) ────────────
    def _tier(self, tier: str) -> str:
        if self.model_mode == "custom":
            return getattr(self, f"custom_{tier}_model")
        return MODE_MODELS[self.model_mode][tier]

    @property
    def chat_model(self) -> str:
        """The everyday model (CHAT tier)."""
        return self._tier("chat")

    @property
    def smart_model(self) -> str:
        """The careful marking model (SMART tier)."""
        return self._tier("smart")

    @property
    def vision_model(self) -> str:
        """The handwriting model (VISION tier, must support images)."""
        return self._tier("vision")

    @property
    def embedding_model(self) -> str:
        """The embedding model."""
        return self._tier("embedding")

    @property
    def grading_model(self) -> str:
        """The model used for grading (the SMART tier; the name the course uses)."""
        return self.smart_model

    @property
    def transcription_model(self) -> str:
        """The model used to read handwriting (the VISION tier)."""
        return self.vision_model

    @property
    def models(self) -> dict[str, str]:
        """{"chat", "smart", "vision", "embedding"} → "provider:model", resolved for the current mode."""
        return {tier: self._tier(tier) for tier in ("chat", "smart", "vision", "embedding")}

    @property
    def provider(self) -> str:
        """The provider part of chat_model, e.g. 'openai'."""
        return provider_of(self.chat_model)

    @property
    def required_keys(self) -> list[str]:
        """The API-key variables this mode needs, e.g. ["OPENROUTER_API_KEY"] (empty for all-Ollama)."""
        keys: list[str] = []
        for model in self.models.values():
            env = PROVIDER_KEYS.get(provider_of(model))
            if env and env not in keys:
                keys.append(env)
        return keys

    @property
    def is_free(self) -> bool:
        """True in MODEL_MODE=free (rate-limited free models: fewer parallel calls, 429 retries)."""
        return self.model_mode == "free"

    def max_concurrency(self, default: int) -> int:
        """How many model calls to run at once: `default`, but at most 2 in free mode (rate limits)."""
        return min(default, FREE_MAX_CONCURRENCY) if self.is_free else default

    def describe_models(self) -> str:
        """One line for banners, e.g. '🧠 MODEL_MODE=cheap: gpt-6-luna (chat · grading · vision) · ...'."""
        m = self.models

        def short(model: str) -> str:
            return model if self.model_mode == "custom" else model.split(":", 1)[-1]

        tiers = {"chat": m["chat"], "grading": m["smart"], "vision": m["vision"]}
        if len(set(tiers.values())) == 1:
            chat_part = f"{short(m['chat'])} (chat · grading · vision)"
            if m["chat"].endswith(FREE_ROUTER):
                chat_part = f"{short(m['chat'])} (the router picks a free model for each call)"
        else:
            chat_part = " · ".join(f"{name} {short(model)}" for name, model in tiers.items())
        price = {"cheap": " · $0.10 / $0.50 per 1M tokens", "free": " · $0 (rate-limited)"}
        return (
            f"🧠 MODEL_MODE={self.model_mode}: {chat_part} · embeddings {short(m['embedding'])}"
            f"{price.get(self.model_mode, '')}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once. Call `get_settings.cache_clear()` after editing .env."""
    return Settings()


settings = get_settings()


def max_concurrency(default: int) -> int:
    """`settings.max_concurrency(default)`, read at call time (so tests and notebooks can switch modes)."""
    return settings.max_concurrency(default)
