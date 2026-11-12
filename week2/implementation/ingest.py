"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 2 · Day 5: Ingestion, the SIMPLE version                    │
│                                                                      │
│  What this file does                                                 │
│    The shortest honest RAG ingestion: read each PDF's text, cut it   │
│    into fixed-size chunks, embed them, save them in Chroma.          │
│    No header stripping, no headings, no hashing: every run wipes     │
│    and rebuilds. Compare it with ../pro_implementation/ingest.py.    │
│                                                                      │
│  Run : uv run week2/implementation/ingest.py                         │
│  ⏮ Built on: week2/day2–3.ipynb   ⏭ Used by: answer.py (this folder)│
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import sys
from pathlib import Path

# 💡 Lets `uv run week2/implementation/ingest.py` find the gradewise package from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pymupdf  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from gradewise import config, datasets  # noqa: E402
from gradewise.console import banner, step, success  # noqa: E402

KNOWLEDGE_BASE = config.DATA_DIR / "knowledge-base"
COLLECTION = "cs201_simple"


def db_dir() -> Path:
    return config.OUTPUT_DIR / "vector_db_simple"  # 💡 its own folder, so it never touches the pro store


def fetch_documents() -> list[Document]:
    """One Document per PDF page, with just the file name and page number as metadata."""
    documents = []
    for pdf in datasets.kb_files(KNOWLEDGE_BASE):  # 💡 all 4 PDFs, or unit1 + the handbook with DATA_SIZE=small
        with pymupdf.open(pdf) as book:
            for number, page in enumerate(book, start=1):
                documents.append(Document(page.get_text(), metadata={"source": pdf.name, "page": number}))
    return documents


def create_chunks(documents: list[Document]) -> list[Document]:
    # ⚠️ Fixed-size chunks ignore the document's structure: a chunk can start in the middle
    #    of "1.5 Synchronisation" and end in "1.6 Deadlocks". The pro version fixes this.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    return splitter.split_documents(documents)


def create_embeddings(chunks: list[Document], embeddings=None) -> Chroma:
    """Delete the old collection (if any) and embed every chunk again."""
    if embeddings is None:
        from gradewise.models import embeddings as default_embeddings

        embeddings = default_embeddings()
    store = Chroma(collection_name=COLLECTION, embedding_function=embeddings, persist_directory=str(db_dir()))
    store.reset_collection()  # 💡 simple but wasteful: we pay to re-embed everything, every time
    store.add_documents(chunks)
    return store


def ingest(embeddings=None) -> int:
    """Run the whole simple pipeline and return the number of chunks stored."""
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks, embeddings)
    return len(chunks)


if __name__ == "__main__":
    banner("Week 2 · Simple ingestion", "load → fixed-size chunks → embed → Chroma")
    step("Reading PDFs, chunking and embedding…")
    count = ingest()
    success(
        "Simple ingestion complete",
        {"Chunks": count, "Collection": COLLECTION, "Stored in": db_dir()},
        next_step="uv run week2/implementation/answer.py",
    )
