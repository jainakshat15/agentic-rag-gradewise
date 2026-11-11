"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 2 · Day 2: Loading the course PDFs                          │
│                                                                      │
│  What this file does                                                 │
│    Turns each PDF page into a LangChain `Document` of clean          │
│    Markdown (via pymupdf4llm), with the metadata every later stage   │
│    relies on: source, page (1-based), title, unit and section.       │
│    Running headers, footers and "Page X of Y" lines are stripped.    │
│                                                                      │
│  Use it                                                              │
│    from gradewise.ingestion.loaders import load_pdf                  │
│    pages = load_pdf(DATA_DIR / "knowledge-base/unit1_...pdf")        │
│                                                                      │
│  ⏮ Built on: week2/day2.ipynb   ⏭ Used by: chunking.py, pipeline.py │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

from gradewise.errors import DataNotFoundError

# 💡 "Page 3 of 7", "page 3/7", "- 3 -": page numbers are noise for retrieval.
_PAGE_NUMBER = re.compile(r"^\s*(?:-?\s*page\s+\d+\s*(?:of|/)\s*\d+\s*-?|-\s*\d+\s*-)\s*$", re.IGNORECASE)
_PAGE_NUMBER_INLINE = re.compile(r"\s*\bpage\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_UNIT = re.compile(r"unit\s*(\d+)", re.IGNORECASE)


def clean_heading(text: str) -> str:
    """'## **1.6 Deadlocks**' → '1.6 Deadlocks' (no hashes, no bold markers)."""
    text = _HEADING.sub(r"\2", text.strip())
    text = text.replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", text).strip()


def _signature(line: str) -> str:
    # 💡 Running headers repeat with tiny changes (the page number), so we compare
    #    lines with digits blanked out: "Page 3 of 7" and "Page 4 of 7" look the same.
    return re.sub(r"\d+", "#", line.strip().lower())


def find_running_lines(pages: list[str], edge: int = 3, min_share: float = 0.6) -> set[str]:
    """Find header/footer lines: lines near the top/bottom of a page that repeat on most pages."""
    if len(pages) < 2:
        return set()
    counts: Counter[str] = Counter()
    for text in pages:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        # ⚠️ Never treat a heading as a running line: "## 1.3 Threads" and "## 1.4 CPU…"
        #    don't match, but a short heading repeated on each page could.
        candidates = [ln for ln in lines[:edge] + lines[-edge:] if not ln.lstrip().startswith("#")]
        counts.update({_signature(ln) for ln in candidates})
    threshold = max(2, round(min_share * len(pages)))
    return {sig for sig, n in counts.items() if n >= threshold}


def strip_running_lines(text: str, running: set[str]) -> str:
    """Remove repeated headers/footers and page-number lines from one page of Markdown."""
    kept = []
    for line in text.splitlines():
        if _signature(line) in running or _PAGE_NUMBER.match(line):
            continue
        kept.append(_PAGE_NUMBER_INLINE.sub("", line) if _PAGE_NUMBER_INLINE.search(line) else line)
    text = "\n".join(kept)
    # ⚠️ pymupdf4llm sometimes puts a blank line between two halves of one wrapped
    #    sentence ("...the operating system\n\nthat runs programs"). We re-join them so
    #    a chunk boundary never lands mid-sentence just because of the PDF layout.
    text = re.sub(r"(?<=[a-z,;])\n\n(?=[a-z(])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _unit_from(meta: dict, path: Path) -> int:
    for text in (meta.get("keywords", ""), path.stem, meta.get("title", "")):
        match = _UNIT.search(text or "")
        if match:
            return int(match.group(1))
    return 0  # 💡 unit 0 = the course handbook (general, not tied to one unit)


def load_pdf(path: Path | str) -> list[Document]:
    """Load one PDF as one clean Markdown `Document` per page."""
    import pymupdf4llm

    path = Path(path)
    if not path.exists():
        raise DataNotFoundError(f"Can't find the PDF {path}.")
    raw_pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    texts = [p["text"] for p in raw_pages]
    running = find_running_lines(texts)
    meta0 = raw_pages[0]["metadata"] if raw_pages else {}
    title = (meta0.get("title") or path.stem.replace("_", " ")).strip()
    unit = _unit_from(meta0, path)

    docs: list[Document] = []
    section = ""  # 💡 carried across pages: page 5 may continue the section that began on page 4
    for number, text in enumerate(texts, start=1):
        clean = strip_running_lines(text, running)
        section_at_start = section
        for line in clean.splitlines():
            if _HEADING.match(line) and line.startswith("##"):
                section = clean_heading(line)
        if not clean:
            continue
        docs.append(
            Document(
                page_content=clean,
                metadata={
                    "source": path.name,
                    "page": number,  # ⚠️ 1-based, like the page numbers a human reads
                    "title": title,
                    "unit": unit,
                    "section": section_at_start,  # the section in force where the page starts
                },
            )
        )
    return docs


def load_directory(kb_dir: Path | str) -> list[Document]:
    """Load every PDF in a folder (sorted by name, so runs are reproducible)."""
    kb_dir = Path(kb_dir)
    pdfs = sorted(kb_dir.glob("*.pdf"))
    if not pdfs:
        raise DataNotFoundError(f"No PDFs found in {kb_dir}.")
    return [doc for pdf in pdfs for doc in load_pdf(pdf)]
