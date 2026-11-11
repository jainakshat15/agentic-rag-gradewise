"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 2 · Day 3: A persistent Chroma store                        │
│                                                                      │
│  What this file does                                                 │
│    Opens (or creates) the Chroma vector store that lives on disk in  │
│    output/vector_db. One collection per course: "cs201" by default.  │
│                                                                      │
│  Use it                                                              │
│    from gradewise.ingestion.store import get_vectorstore             │
│    store = get_vectorstore("cs201")                                  │
│                                                                      │
│  ⏮ Built on: week2/day3.ipynb   ⏭ Used by: pipeline.py, rag/        │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gradewise import config

DEFAULT_COLLECTION = "cs201"


def vector_db_dir() -> Path:
    """Where the vector store lives. Read at call time, so tests can point OUTPUT_DIR elsewhere."""
    return config.OUTPUT_DIR / "vector_db"


def get_vectorstore(
    collection: str = DEFAULT_COLLECTION,
    embeddings: Any | None = None,
    persist_dir: Path | str | None = None,
):
    """Open the persistent Chroma collection `collection` (created on first use).

    💡 `embeddings` defaults to `gradewise.models.embeddings()` (your EMBEDDING_MODEL).
       Pass any LangChain `Embeddings`, e.g. `DeterministicFakeEmbedding(size=64)` in tests.
    ⚠️ Always query a collection with the SAME embedding model that built it.
       Vectors from different models live in different "spaces" and can't be compared.
    """
    from langchain_chroma import Chroma

    if embeddings is None:
        from gradewise.models import embeddings as default_embeddings

        embeddings = default_embeddings()
    directory = Path(persist_dir) if persist_dir else vector_db_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(directory),
        collection_metadata={"hnsw:space": "cosine"},  # 💡 cosine distance suits text embeddings
    )
