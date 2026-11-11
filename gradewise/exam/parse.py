"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 1 · Day 5: Reading the exam pack                            │
│                                                                      │
│  What this file does                                                 │
│    Turns a PDF into clean Markdown with pymupdf4llm, page by page,   │
│    and strips the headers and footers that repeat on every page      │
│    ("Page 3 of 7", the school name...). The LLM then reads only the  │
│    content that matters.                                             │
│                                                                      │
│  Use it                                                              │
│    from gradewise.exam.parse import parse_pdf                        │
│    doc = parse_pdf("data/exams/cs_midterm/question_paper.pdf")       │
│    print(doc.page_count, doc.text[:500])                             │
│                                                                      │
│  ⏮ Built on: week1/day4.ipynb (reading PDFs)                         │
│  ⏭ Used by: gradewise/exam/extract.py, gradewise/exam/graph.py       │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from gradewise.errors import DataNotFoundError

# 💡 Headers and footers live at the very top and bottom of a page. We only look at
#    this many non-empty lines at each edge, so a sentence that happens to repeat in
#    the middle of the notes is never removed.
EDGE_LINES = 3

# 💡 Running lines we KNOW our Northbridge PDFs print. They are stripped from a page edge even
#    when they appear only once (e.g. a "(turn over)" footer that differs on the last page).
#    Patterns are matched against the line's signature: lower case, markdown removed.
KNOWN_RUNNING_LINES: tuple[str, ...] = (
    r"northbridge institute · school of computing",
    r"\bpage (\d+|#) of \d+\b",  # "Page 1 of 2", "… Page 1 of 2 (turn over)" (# = this page's number)
    r"^cs201 systems fundamentals · ",  # the running header; NOT the "CS201 …: Midterm" title
)
# ⚠️ Never strip an exam heading, whatever a pattern says.
_PROTECTED = re.compile(r"^(question|q)\s*\d+\b")


@dataclass
class ParsedPage:
    """One page of a PDF, as Markdown, with its 1-based page number."""

    number: int
    text: str


@dataclass
class ParsedDocument:
    """A whole PDF as clean Markdown, plus what we stripped (handy for debugging)."""

    path: Path
    pages: list[ParsedPage]
    stripped_lines: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        """All pages joined with a blank line: what we send to the LLM."""
        return "\n\n".join(p.text for p in self.pages if p.text)

    def with_page_markers(self) -> str:
        """Like `.text`, but each page starts with `<!-- page N -->`, so a model can cite pages."""
        return "\n\n".join(f"<!-- page {p.number} -->\n{p.text}" for p in self.pages if p.text)


def _signature(line: str, page_number: int) -> str:
    """A line without markdown decoration, and with *this page's number* blanked out.

    So "Page 3 of 7" (on page 3) and "Page 4 of 7" (on page 4) match.
    ⚠️ We blank the number only in lines that look like page numbering ("Page 3", "3 / 7", "- 3 -"),
       and only once. Blank every digit and "Question 3" / "Question 4" at the top of two pages
       would look like a repeated header and vanish from the exam!
    """
    line = re.sub(r"\s+", " ", re.sub(r"[*_#`>|]", "", line)).strip().lower()
    looks_like_page_number = "page" in line or re.fullmatch(r"[-–\s]*\d+\s*((of|/)\s*\d+)?[-–\s]*", line)
    if looks_like_page_number:
        line = re.sub(rf"(?<!\d){page_number}(?!\d)", "#", line, count=1)
    return line


def _edge_indexes(lines: list[str], edge: int) -> list[int]:
    """Indexes of the first and last `edge` non-empty lines of a page."""
    non_empty = [i for i, line in enumerate(lines) if line.strip()]
    return sorted(set(non_empty[:edge] + non_empty[-edge:]))


def _is_known_running_line(signature: str, patterns: Sequence[str]) -> bool:
    """True if this signature matches one of the known header/footer patterns (and isn't a heading)."""
    if not signature or _PROTECTED.match(signature):
        return False
    return any(re.search(pattern, signature) for pattern in patterns)


def strip_repeated_lines(
    pages: list[str],
    *,
    edge: int = EDGE_LINES,
    min_share: float = 0.5,
    known_lines: Sequence[str] = KNOWN_RUNNING_LINES,
) -> tuple[list[str], list[str]]:
    """Remove lines that repeat at the top or bottom of many pages.

    A line counts as a header/footer when (ignoring the page number) it appears at a page edge on
    at least `min_share` of the pages, and on at least 2 pages. Edge lines that match one of the
    `known_lines` regexes (see KNOWN_RUNNING_LINES) go too, even if they appear only once.
    Returns `(clean_pages, stripped_signatures)`.

    ⚠️ A one-page PDF has nothing to compare against, so only the known lines are stripped.
    """
    split = [p.splitlines() for p in pages]
    repeated: set[str] = set()
    if len(pages) >= 2:
        counts: Counter[str] = Counter()
        for number, lines in enumerate(split, start=1):
            # 💡 count each signature once per page, or a doubled footer would cheat
            counts.update({_signature(lines[i], number) for i in _edge_indexes(lines, edge)} - {""})
        threshold = max(2, round(min_share * len(pages)))
        repeated = {sig for sig, n in counts.items() if n >= threshold}

    clean: list[str] = []
    stripped = set(repeated)
    for number, lines in enumerate(split, start=1):
        drop: set[int] = set()
        for i in _edge_indexes(lines, edge):
            sig = _signature(lines[i], number)
            if sig in repeated or _is_known_running_line(sig, known_lines):
                drop.add(i)
                stripped.add(sig)
        kept = [line for i, line in enumerate(lines) if i not in drop]
        text = "\n".join(kept)
        clean.append(re.sub(r"\n{3,}", "\n\n", text).strip())  # tidy the gaps we left
    return clean, sorted(stripped)


def parse_pdf(path: str | Path, *, strip_headers: bool = True) -> ParsedDocument:
    """Read a PDF into clean, per-page Markdown.

    💡 pymupdf4llm keeps headings (`#`), bold, lists and tables as Markdown, which LLMs
       read far better than the raw text soup you get from plain `page.get_text()`.
    """
    import pymupdf4llm  # 💡 imported here so `import gradewise.exam` stays fast

    path = Path(path)
    if not path.exists():
        raise DataNotFoundError(f"Can't find the PDF {path}.")

    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    texts = [chunk["text"] for chunk in chunks]
    stripped: list[str] = []
    if strip_headers:
        texts, stripped = strip_repeated_lines(texts)
    pages = [ParsedPage(number=i + 1, text=t) for i, t in enumerate(texts)]
    return ParsedDocument(path=path, pages=pages, stripped_lines=stripped)


def read_document(source: str | Path | ParsedDocument) -> str:
    """Get plain text from whatever you have: a PDF path, a .md/.txt path, a parsed doc, or text itself.

    ⚠️ A `str` is treated as a *path* only if it looks like one (one line, ends in .pdf/.md/.txt
       and the file exists). Anything else is treated as the document's text.
    """
    if isinstance(source, ParsedDocument):
        return source.text
    if isinstance(source, str):
        looks_like_path = "\n" not in source and source.lower().endswith((".pdf", ".md", ".txt"))
        if not (looks_like_path and Path(source).exists()):
            return source
        source = Path(source)
    if source.suffix.lower() == ".pdf":
        return parse_pdf(source).text
    if not source.exists():
        raise DataNotFoundError(f"Can't find the file {source}.")
    return source.read_text(encoding="utf-8")
