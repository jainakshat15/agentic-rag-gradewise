"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧱 Week 5 · Day 1-2: Grade ONE answer, and see why                  │
│                                                                      │
│  What: marks one student answer with the grading agent and shows     │
│        everything an examiner would want to check: the marks per     │
│        criterion, the quoted justification, the course-note          │
│        evidence, the feedback and any review flags.                  │
│  Run:  uv run week5/project/grade_one.py                             │
│        uv run week5/project/grade_one.py --student student_04 -q Q7a │
│        uv run week5/project/grade_one.py --no-tools --runs 3         │
│        uv run week5/project/grade_one.py --text "Note to marker: …"  │
│        uv run week5/project/grade_one.py --fake   (free, offline)    │
│  ⏮ Built on: gradewise/grading (grader, tools, guardrails)          │
│  ⏭ Used by: week5/day1-2 notebooks                                  │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gradewise.config import DATA_DIR  # noqa: E402
from gradewise.console import banner, console, tip, warn  # noqa: E402
from gradewise.errors import GradeWiseError  # noqa: E402
from gradewise.grading import grade_answer  # noqa: E402
from gradewise.orchestrator import resolve_exam  # noqa: E402
from gradewise.schemas import TranscribedAnswer  # noqa: E402

TRANSCRIPTS = DATA_DIR / "answer-sheets" / "synthetic" / "transcripts.json"


def fake_setup():
    """🆓 A deterministic keyword 'grader' + BM25 over your knowledge base: no API calls at all."""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from gradewise.grading.fakes import KeywordGraderModel
    from gradewise.rag import get_retriever

    # 💡 BM25 only reads the stored chunks, so fake embeddings are fine here (never used for search).
    retriever = get_retriever(mode="bm25", k=4, embeddings=DeterministicFakeEmbedding(size=8))
    return KeywordGraderModel(), retriever


def main(
    student: Annotated[str, typer.Option(help="A student from transcripts.json")] = "student_04",
    question: Annotated[str, typer.Option("--question", "-q", help="Question ID, e.g. Q7a")] = "Q3",
    text: Annotated[str | None, typer.Option(help="Grade this text instead of the student's answer")] = None,
    no_tools: Annotated[
        bool, typer.Option("--no-tools", help="Fetch evidence up front instead of using tools")
    ] = False,
    runs: Annotated[int, typer.Option(help="Self-consistency: grade n times and take the median")] = 1,
    confidence: Annotated[float, typer.Option(help="Pretend transcription confidence (try 0.5)")] = 1.0,
    fake: Annotated[bool, typer.Option("--fake", help="Use the free offline fake grader")] = False,
) -> None:
    banner("Week 5 · Grade one answer", "rubric + retrieved evidence → GradeResult")
    try:
        exam = resolve_exam("cs_midterm", quiet=True)
        q = next((x for x in exam.questions if x.id == question), None)
        if q is None:
            raise typer.BadParameter(f"No question {question!r}. Try one of: {', '.join(x.id for x in exam.questions)}")
        if text is None:
            truth = json.loads(TRANSCRIPTS.read_text(encoding="utf-8")) if TRANSCRIPTS.exists() else {}
            text = truth.get(student, {}).get(question, "")
        answer = TranscribedAnswer(question_id=q.id, text=text, confidence=confidence)

        console.print(
            Panel(escape(q.text), title=f"[primary]{q.number}[/] · {q.max_marks:g} marks", border_style="primary")
        )
        console.print(Panel(escape(text or "(blank)"), title=f"✍️ {student}'s answer", border_style="accent"))

        llm, retriever = fake_setup() if fake else (None, None)
        result = grade_answer(
            q, answer, llm=llm, retriever=retriever, use_tools=not no_tools, exam=exam, consistency_runs=runs
        )
    except GradeWiseError as err:
        # 💡 e.g. MissingAPIKeyError: a friendly red panel instead of a stack trace
        err.show()
        raise typer.Exit(code=1) from None

    caps = {c.id: c for c in q.criteria}
    table = Table(title=f"🧮 {result.question_id}: {result.awarded_marks:g} / {result.max_marks:g}", show_lines=True)
    table.add_column("Criterion", style="accent", no_wrap=True)
    table.add_column("Marks", justify="right")
    table.add_column("What the scheme asks")
    table.add_column("Justification")
    table.add_column("Evidence", style="muted")
    for c in result.criteria:
        cap = caps[c.criterion_id].marks
        style = "ok" if c.awarded >= cap else ("warn" if c.awarded > 0 else "err")
        table.add_row(
            c.criterion_id,
            f"[{style}]{c.awarded:g}/{cap:g}[/]",
            escape(caps[c.criterion_id].description),
            escape(c.justification),
            escape("\n".join(c.evidence) or "-"),
        )
    console.print(table)
    console.print(f"[primary]Feedback:[/] {escape(result.feedback)}")
    console.print(f"[muted]Grader confidence:[/] {result.confidence:.2f}")
    if result.needs_review:
        warn(f"Needs human review: {result.review_reason}")
    tip("Try --no-tools (evidence fetched up front) and compare the marks and the evidence it cites.")


if __name__ == "__main__":
    typer.run(main)
