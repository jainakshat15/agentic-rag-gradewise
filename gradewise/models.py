"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 1 · Day 2: One place to create models                       │
│                                                                      │
│  What this file does                                                 │
│    Wraps LangChain's `init_chat_model` so every notebook gets its    │
│    models the same way, with a clear error if a key is missing.      │
│                                                                      │
│      chat_model()     → everyday, cheap model                        │
│      grading_model()  → the careful model used for marking (week 5)  │
│      vision_model()   → reads handwriting (week 4)                   │
│      embeddings()     → turns text into vectors (week 2)             │
│                                                                      │
│  🔁 RECAP: model names come from gradewise/config.py (your .env)     │
│  ⏭ Used by: every notebook from week1/day2.ipynb onwards            │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import os
from typing import Any

from gradewise.config import settings
from gradewise.errors import InvalidModelError, MissingAPIKeyError

# 💡 Which environment variable each provider needs. Ollama runs locally: no key.
_PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "ollama": None,
}


def _check(model: str) -> str:
    """Validate a "provider:model" string and make sure its API key is available."""
    if ":" not in model:
        raise InvalidModelError(f'"{model}" has no provider prefix.')
    provider = model.split(":", 1)[0]
    env_var = _PROVIDER_KEYS.get(provider, "")
    if env_var:
        # ⚠️ LangChain reads keys from the environment, so we copy them out of .env.
        secret = getattr(settings, env_var.lower(), None)
        if secret is not None and not os.getenv(env_var):
            os.environ[env_var] = secret.get_secret_value()
        if not os.getenv(env_var):
            raise MissingAPIKeyError(f"{env_var} is not set, but your model is {model}.")
    return model


def make_chat_model(model: str, **kwargs: Any):
    """Create any chat model from a "provider:model" string."""
    from langchain.chat_models import init_chat_model

    return init_chat_model(_check(model), **kwargs)


def chat_model(**kwargs: Any):
    """The everyday model (CHAT_MODEL in .env)."""
    return make_chat_model(settings.chat_model, **kwargs)


def grading_model(**kwargs: Any):
    """The careful marking model (SMART_MODEL, or CHAT_MODEL in budget mode)."""
    return make_chat_model(settings.grading_model, **kwargs)


def vision_model(**kwargs: Any):
    """The handwriting model (VISION_MODEL, or CHAT_MODEL in budget mode)."""
    return make_chat_model(settings.transcription_model, **kwargs)


def embeddings(**kwargs: Any):
    """The embedding model (EMBEDDING_MODEL in .env)."""
    from langchain.embeddings import init_embeddings

    return init_embeddings(_check(settings.embedding_model), **kwargs)
