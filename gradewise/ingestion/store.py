"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 2 · Day 3: A persistent Chroma store                        │
│                                                                      │
│  What this file does                                                 │
│    Opens (or creates) the Chroma vector store that lives on disk in  │
│    output/vector_db. One collection per course: "cs201" by default.  │
│                                                                      │
│  🔢 Each EMBEDDING MODEL gets its own sub-folder, e.g.               │
│       output/vector_db/openrouter_openai_text-embedding-3-small/     │
│       output/vector_db/openrouter_nvidia_nemotron-3.5-embed-1b_free/ │
│     so switching MODEL_MODE never mixes vectors of different sizes.  │
│                                                                      │
│  Use it                                                              │
│    from gradewise.ingestion.store import get_vectorstore             │
│    store = get_vectorstore("cs201")                                  │
│                                                                      │
│  ⏮ Built on: week2/day3.ipynb   ⏭ Used by: pipeline.py, rag/        │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from gradewise import config

DEFAULT_COLLECTION = "cs201"

# 💡 Which provider an Embeddings class belongs to, from the package it comes from.
_PROVIDER_BY_PACKAGE = {
    "langchain_openai": "openai",
    "langchain_google_genai": "google_genai",
    "langchain_ollama": "ollama",
    "langchain_anthropic": "anthropic",
}


def vector_db_dir() -> Path:
    """The ROOT of the vector stores. Read at call time, so tests can point OUTPUT_DIR elsewhere.

    💡 The store for your embedding model is a sub-folder: see `store_dir()`.
    """
    return config.OUTPUT_DIR / "vector_db"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "default"


def embedding_key(embeddings: Any | None = None, model: str | None = None) -> str:
    """A folder-safe name for an embedding model: 'openai_text-embedding-3-small'.

    Pass a "provider:model" string (`model`), an Embeddings object, or nothing (your EMBEDDING_MODEL).
    """
    if model is None and embeddings is None:
        model = config.get_settings().embedding_model
    if model is not None:
        provider, _, name = model.partition(":") if ":" in model else ("openai", "", model)
        return _slug(f"{provider}_{name.removeprefix('models/')}")
    inner = getattr(embeddings, "inner", embeddings)  # 💡 unwrap gradewise.models.RetryingEmbeddings
    name = getattr(inner, "model", None) or getattr(inner, "model_name", None)
    if name:
        package = type(inner).__module__.split(".")[0]
        provider = _PROVIDER_BY_PACKAGE.get(package, package)
        if "openrouter.ai" in str(getattr(inner, "openai_api_base", None) or ""):
            provider = "openrouter"  # 💡 OpenAIEmbeddings pointed at OpenRouter (see models.openrouter_embeddings)
        return _slug(f"{provider}_{str(name).removeprefix('models/')}")
    size = getattr(inner, "size", None)  # 🧪 the fake embeddings used by tests and demo mode
    return _slug(f"{type(inner).__name__}_{size}" if size else type(inner).__name__)


def store_dir(persist_dir: Path | str | None = None, embeddings: Any | None = None, model: str | None = None) -> Path:
    """The folder holding the collections for ONE embedding model: <persist_dir or output/vector_db>/<key>."""
    root = Path(persist_dir) if persist_dir else vector_db_dir()
    return root / embedding_key(embeddings, model)


def get_vectorstore(
    collection: str = DEFAULT_COLLECTION,
    embeddings: Any | None = None,
    persist_dir: Path | str | None = None,
    *,
    embedding_model: str | None = None,
):
    """Open the persistent Chroma collection `collection` (created on first use).

    💡 `embeddings` defaults to `gradewise.models.embeddings()` (your MODEL_MODE's embedding model).
       Pass any LangChain `Embeddings`, e.g. `DeterministicFakeEmbedding(size=64)` in tests.
    💡 The store lives in a sub-folder named after the embedding model (see `store_dir`), so each
       model has its own collections. `embedding_model="provider:model"` picks the folder explicitly.
    ⚠️ Always query a collection with the SAME embedding model that built it.
       Vectors from different models live in different "spaces" and can't be compared.
    """
    from langchain_chroma import Chroma

    if embeddings is None:
        from gradewise.models import embeddings as default_embeddings

        embeddings = default_embeddings()
        embedding_model = embedding_model or config.get_settings().embedding_model
    directory = store_dir(persist_dir, embeddings, embedding_model)
    directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(directory),
        collection_metadata={"hnsw:space": "cosine"},  # 💡 cosine distance suits text embeddings
    )
