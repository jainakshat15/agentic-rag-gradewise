"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 2 · Day 5: The retrieval test set                           │
│                                                                      │
│  What this file does                                                 │
│    Loads `tests.jsonl` (one question per line) and validates every   │
│    line with Pydantic, so a typo in the test set fails loudly here   │
│    instead of quietly skewing your MRR.                              │
│                                                                      │
│  Run : uv run week2/evaluation/test.py                               │
│  ⏮ Built on: week1 Pydantic   ⏭ Used by: eval.py (this folder)      │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, Field, ValidationError  # noqa: E402

from gradewise.config import DATA_DIR  # noqa: E402
from gradewise.console import console, success, warn  # noqa: E402
from gradewise.errors import DataNotFoundError, GradeWiseError  # noqa: E402

TESTS_FILE = Path(__file__).parent / "tests.jsonl"
Category = Literal["direct_fact", "numerical", "comparison", "spanning", "holistic"]


class InvalidTestSetError(GradeWiseError):
    title = "tests.jsonl has a problem"
    cause = "A line isn't valid JSON, or is missing a field / has a wrong category."
    fix = 'Each line needs "question", "keywords", "reference_answer", "source" and "category".'
    help_ref = "week2/evaluation/README.md"


class EvalQuestion(BaseModel):
    """One line of tests.jsonl: a question, and how we'll know retrieval found the answer."""

    question: str = Field(min_length=3)
    keywords: list[str] = Field(min_length=1, description="Words that MUST appear in a relevant chunk")
    reference_answer: str = Field(min_length=1, description="The answer a careful marker would accept")
    source: str = Field(description="The PDF the answer comes from, e.g. unit1_operating_systems.pdf")
    category: Category


def load_tests(path: Path | str = TESTS_FILE) -> list[EvalQuestion]:
    """Read and validate every line. Blank lines are allowed; anything else must be valid."""
    path = Path(path)
    if not path.exists():
        raise DataNotFoundError(f"Can't find the test set {path}.", fix="Run `git pull` to get tests.jsonl.")
    tests = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            tests.append(EvalQuestion.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise InvalidTestSetError(f"Line {number} of {path.name}: {exc}") from exc
    return tests


def check_sources(tests: list[EvalQuestion], kb_dir: Path = DATA_DIR / "knowledge-base") -> list[str]:
    """Sources named in the tests that aren't in the knowledge base (should be none)."""
    available = {p.name for p in kb_dir.glob("*.pdf")}
    return sorted({t.source for t in tests} - available)


if __name__ == "__main__":
    try:
        tests = load_tests(sys.argv[1] if len(sys.argv) > 1 else TESTS_FILE)
    except GradeWiseError as err:
        err.show()
        sys.exit(1)
    missing = check_sources(tests)
    for source in missing:
        warn(f"Source {source} isn't in data/knowledge-base/")
    categories = Counter(t.category for t in tests)
    sources = Counter(t.source for t in tests)
    console.print()
    success(
        f"{len(tests)} test questions are valid",
        {
            **{f"Category · {c}": n for c, n in sorted(categories.items())},
            **{f"Source · {s}": n for s, n in sorted(sources.items())},
        },
        next_step="uv run week2/evaluation/eval.py",
    )
    sys.exit(1 if missing else 0)
