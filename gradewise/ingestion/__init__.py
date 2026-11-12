"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 2: GradeWise's knowledge base (ingestion)                   │
│                                                                      │
│  What this package does                                              │
│      loaders.py   → PDF pages → clean Markdown Documents   (day 2)   │
│      chunking.py  → structure-aware chunks + metadata      (day 2)   │
│      store.py     → persistent Chroma at output/vector_db  (day 3)   │
│      pipeline.py  → idempotent `ingest()`                  (day 5)   │
│                                                                      │
│  Run: uv run week2/pro_implementation/ingest.py                      │
│  ⏭ Used by: gradewise/rag, week4/project, week5 grading             │
╰──────────────────────────────────────────────────────────────────────╯
"""

from gradewise.ingestion.chunking import chunk_documents
from gradewise.ingestion.loaders import load_directory, load_pdf
from gradewise.ingestion.pipeline import IngestReport, ingest
from gradewise.ingestion.store import embedding_key, get_vectorstore, store_dir, vector_db_dir

__all__ = [
    "IngestReport",
    "chunk_documents",
    "embedding_key",
    "get_vectorstore",
    "ingest",
    "load_directory",
    "load_pdf",
    "store_dir",
    "vector_db_dir",
]
