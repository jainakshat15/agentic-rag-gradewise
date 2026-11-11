"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 2 · Day 5: The GradeWise ingestion pipeline                 │
│                                                                      │
│  What this file does                                                 │
│    PDFs → clean pages → structure-aware chunks → Chroma, and it is   │
│    IDEMPOTENT: each file's SHA-256 hash is stored with its chunks,   │
│    so re-running skips unchanged files and re-indexes changed ones.  │
│                                                                      │
│  Run                                                                 │
│    uv run week2/pro_implementation/ingest.py                         │
│    (or, in code)  from gradewise.ingestion import ingest; ingest()   │
│                                                                      │
│  ⏮ Built on: week2/day3.ipynb   ⏭ Used by: rag/, week4, week5       │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gradewise.config import DATA_DIR, settings
from gradewise.console import step, success, tip
from gradewise.errors import DataNotFoundError
from gradewise.ingestion.chunking import chunk_documents
from gradewise.ingestion.loaders import load_pdf
from gradewise.ingestion.store import DEFAULT_COLLECTION, get_vectorstore, vector_db_dir

# 💰 US$ per million tokens (Nov 2026 list prices). Unknown models fall back to "small".
EMBEDDING_PRICES = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}
BATCH_SIZE = 100  # ⚠️ embedding APIs cap how many texts one request may carry


@dataclass
class IngestReport:
    """What one run of `ingest()` did."""

    collection: str
    files: int = 0
    pages: int = 0
    chunks_added: int = 0
    chunks_total: int = 0
    skipped: list[str] = field(default_factory=list)  # unchanged files
    updated: list[str] = field(default_factory=list)  # changed files, re-indexed
    added: list[str] = field(default_factory=list)  # brand-new files
    removed: list[str] = field(default_factory=list)  # files no longer in the folder
    est_tokens: int = 0
    est_cost_usd: float = 0.0


def file_hash(path: Path) -> str:
    """SHA-256 of the file's bytes: changes if (and only if) the file's content changes."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _stored_hashes(store) -> dict[str, tuple[str, list[str]]]:
    """{source: (file_hash, [chunk ids])} for everything already in the collection."""
    found: dict[str, tuple[str, list[str]]] = {}
    data = store.get(include=["metadatas"])
    for chunk_id, meta in zip(data["ids"], data["metadatas"], strict=True):
        meta = meta or {}
        source = meta.get("source", "?")
        h, ids = found.get(source, (meta.get("file_hash", ""), []))
        ids.append(chunk_id)
        found[source] = (h, ids)
    return found


def estimate_cost(texts: list[str], model: str | None = None) -> tuple[int, float]:
    """Rough embedding cost: ~4 characters per token for English text."""
    tokens = sum(len(t) for t in texts) // 4
    name = (model or settings.embedding_model).split(":", 1)[-1]
    price = EMBEDDING_PRICES.get(name, EMBEDDING_PRICES["text-embedding-3-small"])
    return tokens, tokens / 1_000_000 * price


def ingest(
    kb_dir: Path | str = DATA_DIR / "knowledge-base",
    collection: str = DEFAULT_COLLECTION,
    embeddings: Any | None = None,
    reset: bool = False,
    persist_dir: Path | str | None = None,
    quiet: bool = False,
) -> IngestReport:
    """Ingest every PDF in `kb_dir` into the Chroma collection `collection`.

    Safe to run again and again: unchanged files are skipped, changed files have their
    old chunks deleted before the new ones go in, and deleted files are removed.
    `reset=True` wipes the collection first and rebuilds everything.
    """
    kb_dir = Path(kb_dir)
    pdfs = sorted(kb_dir.glob("*.pdf"))
    if not pdfs:
        raise DataNotFoundError(f"No PDFs found in {kb_dir}.")
    say = (lambda _msg: None) if quiet else step

    store = get_vectorstore(collection, embeddings=embeddings, persist_dir=persist_dir)
    if reset:
        say(f"Resetting collection '{collection}'…")
        store.reset_collection()

    report = IngestReport(collection=collection, files=len(pdfs))
    stored = _stored_hashes(store)

    # 🧹 Files that were in the store but are gone from the folder.
    current = {p.name for p in pdfs}
    for source, (_h, ids) in stored.items():
        if source not in current:
            store.delete(ids=ids)
            report.removed.append(source)

    new_texts: list[str] = []
    for pdf in pdfs:
        digest = file_hash(pdf)
        previous = stored.get(pdf.name)
        if previous and previous[0] == digest:
            report.skipped.append(pdf.name)  # 💡 the whole point: no re-embedding, no cost
            continue
        if previous:
            # ⚠️ Delete BEFORE adding. If page 7 was removed from the PDF, its old chunk
            #    would otherwise live on in the store forever, answering questions wrongly.
            store.delete(ids=previous[1])
            report.updated.append(pdf.name)
        else:
            report.added.append(pdf.name)

        say(f"Loading and chunking {pdf.name}…")
        pages = load_pdf(pdf)
        chunks = chunk_documents(pages)
        for chunk in chunks:
            chunk.metadata["file_hash"] = digest
        # 💡 Deterministic IDs: the same chunk always gets the same ID, so a re-run can
        #    never create duplicates even if something above went wrong.
        ids = [f"{pdf.stem}:{digest[:12]}:{c.metadata['chunk_index']}" for c in chunks]
        for start in range(0, len(chunks), BATCH_SIZE):
            store.add_documents(chunks[start : start + BATCH_SIZE], ids=ids[start : start + BATCH_SIZE])
        report.pages += len(pages)
        report.chunks_added += len(chunks)
        new_texts.extend(c.page_content for c in chunks)

    report.chunks_total = len(store.get(include=[])["ids"])
    report.est_tokens, report.est_cost_usd = estimate_cost(new_texts)

    if not quiet:
        success(
            "Ingestion complete",
            {
                "Collection": collection,
                "Files": f"{report.files} ({len(report.added)} new, {len(report.updated)} changed, "
                f"{len(report.skipped)} unchanged, {len(report.removed)} removed)",
                "Pages loaded": report.pages,
                "Chunks added": report.chunks_added,
                "Chunks in store": report.chunks_total,
                "Skipped": ", ".join(report.skipped) or "none",
                "Embedding cost": f"≈ {report.est_tokens:,} tokens ≈ ${report.est_cost_usd:.5f}",
                "Stored in": str(Path(persist_dir) if persist_dir else vector_db_dir()),
            },
            next_step="uv run week2/pro_implementation/answer.py",
        )
        if report.skipped and not report.chunks_added:
            tip("Nothing changed, so nothing was embedded. Edit a PDF (or pass reset=True) to re-index.")
    return report
