"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 2 · Day 5: Ingestion, the PRO version                       │
│                                                                      │
│  What this file does                                                 │
│    The same job as ../implementation/ingest.py, done properly by     │
│    the gradewise package. The pro choices, one per line:             │
│      🚀 pymupdf4llm → Markdown that keeps headings and tables        │
│      🚀 running headers / footers / "Page X of Y" stripped           │
│      🚀 structure-aware chunks (split on ## headings first)          │
│      🚀 rich metadata: source, page, title, unit, section path       │
│      🚀 idempotent: file hashes, unchanged files skipped             │
│                                                                      │
│  Run : uv run week2/pro_implementation/ingest.py [--reset]           │
│  ⏮ Built on: gradewise/ingestion   ⏭ Used by: answer.py, eval.py    │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gradewise.config import DATA_DIR  # noqa: E402
from gradewise.console import banner  # noqa: E402
from gradewise.errors import GradeWiseError  # noqa: E402
from gradewise.ingestion import ingest  # noqa: E402

KNOWLEDGE_BASE = DATA_DIR / "knowledge-base"
COLLECTION = "cs201"  # 💡 one collection per course: a new course is a new collection


def main(reset: bool = False) -> None:
    banner("Week 2 · Pro ingestion", "pymupdf4llm → clean → structure-aware chunks → idempotent Chroma")
    # 💡 Run this twice. The second run embeds nothing and costs nothing: that's idempotency.
    ingest(kb_dir=KNOWLEDGE_BASE, collection=COLLECTION, reset=reset)


if __name__ == "__main__":
    try:
        main(reset="--reset" in sys.argv)
    except GradeWiseError as err:
        err.show()
        sys.exit(1)
