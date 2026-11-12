"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 3 · Day 5: Showing Emma what GradeWise understood           │
│                                                                      │
│  What this file does                                                 │
│    exam_summary(paper)        → a small JSON-able dict (sent with    │
│                                 the interrupt, shown in the UI)      │
│    print_exam_summary(paper)  → a rich table: questions, marks and   │
│                                 criteria counts, problems in yellow  │
│                                                                      │
│  ⏭ Used by: gradewise/exam/graph.py, week3/project/exam_graph.py     │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from typing import Any

from rich.table import Table

from gradewise.console import console, warn
from gradewise.schemas.exam import ExamPaper


def _short(text: str, width: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def exam_summary(paper: ExamPaper) -> dict[str, Any]:
    """A compact view of an exam: enough for a human to approve it at a glance."""
    return {
        "exam_id": paper.exam_id,
        "title": paper.title,
        "subject": paper.subject,
        "total_marks": paper.total_marks,
        "question_count": len(paper.questions),
        "criteria_count": sum(len(q.criteria) for q in paper.questions),
        "questions": [
            {
                "id": q.id,
                "number": q.number,
                "type": q.question_type,
                "max_marks": q.max_marks,
                "criteria": len(q.criteria),
                "criteria_total": q.criteria_total,
                "text": _short(q.text, 80),
            }
            for q in paper.questions
        ],
        "problems": paper.problems(),
    }


def print_exam_summary(paper: ExamPaper, problems: list[str] | None = None) -> None:
    """Pretty-print an ExamPaper as a table. Rows whose criteria don't add up are highlighted."""
    problems = paper.problems() if problems is None else problems
    table = Table(
        title=f"[primary]{paper.title}[/]  [muted]({paper.exam_id} · {paper.subject})[/]",
        header_style="accent",
        show_footer=True,
    )
    criteria_count = sum(len(q.criteria) for q in paper.questions)
    table.add_column("ID", footer="Total", no_wrap=True)
    table.add_column("Type", no_wrap=True)
    table.add_column("Question", footer=f"{len(paper.questions)} questions", overflow="ellipsis")
    table.add_column("Marks", justify="right", no_wrap=True, footer=f"{paper.questions_total:g}/{paper.total_marks:g}")
    table.add_column("Criteria", justify="right", no_wrap=True, footer=str(criteria_count))

    for q in paper.questions:
        ok = abs(q.criteria_total - q.max_marks) <= 0.01
        # 💡 a ⚠ row means the criteria don't add up to the question's marks
        marks = f"{q.max_marks:g}" if ok else f"[warn]⚠ {q.max_marks:g}[/]"
        table.add_row(q.id, q.question_type, _short(q.text, 50), marks, str(len(q.criteria)))

    console.print(table)
    for problem in problems:
        warn(problem)
