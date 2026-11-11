"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧱 Week 5 · Day 4: GradeWise end to end (milestone M5)              │
│                                                                      │
│  What: answer sheets in → report cards out, with Asha in the loop:   │
│        transcribe → review → grade (in parallel) → human review →    │
│        write report cards → store (SQLite + Chroma).                 │
│  Run:  uv run week5/project/orchestrator.py --students 3             │
│        uv run week5/project/orchestrator.py --from-transcripts       │
│        uv run week5/project/orchestrator.py --auto-approve           │
│        uv run week5/project/orchestrator.py --thread batch-1 (resume)│
│        uv run week5/project/orchestrator.py --fake  (free, offline)  │
│  ⏮ Built on: gradewise/orchestrator.py, every week before this one  │
│  ⏭ Next:  uv run week5/project/report_qa.py                         │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.markup import escape
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gradewise.config import DATA_DIR, OUTPUT_DIR, REPO_ROOT  # noqa: E402
from gradewise.console import banner, console, step, success, tip, warn  # noqa: E402
from gradewise.errors import DataNotFoundError, GradeWiseError  # noqa: E402
from gradewise.orchestrator import grade_batch, resolve_exam, sheets_from_transcripts  # noqa: E402

SHEETS_DIR = DATA_DIR / "answer-sheets" / "synthetic"
FAKE_DIR = OUTPUT_DIR / "fake_run"  # 💡 --fake writes here, so fake marks never mix with real ones

ICONS = {
    "load_exam": "📝",
    "transcribe": "✍️ ",
    "review_transcription": "🔎",
    "grade_question": "  ·",
    "grade": "🧮",
    "aggregate": "📊",
    "human_review": "🧑‍🏫",
    "write_report": "📄",
    "store": "💾",
    "paused": "⏸ ",
    "resumed": "▶️ ",
    "done": "✅",
}


def on_event(stage: str, student_id: str | None, detail: dict[str, Any]) -> None:
    """Print each progress event. 💡 The week 6 UI receives exactly these events, as a live stream."""
    who = f"[accent]{student_id}[/] " if student_id else ""
    console.print(f"{ICONS.get(stage, '▸')} [muted]{stage:<20}[/] {who}{escape(detail.get('message', ''))}")


def ask_teacher(payload: dict[str, Any]) -> dict[str, Any]:
    """The human in the loop: show what's flagged, ask for a decision, return the resume value."""
    sid = payload["student_id"]
    table = Table(title=f"🧑‍🏫 {sid}: {payload['type'].replace('_', ' ')}", show_lines=True)
    table.add_column("Question", style="accent")
    if payload["type"] == "human_review":
        table.add_column("Marks", justify="right")
        table.add_column("Why it was flagged")
        for item in payload["items"]:
            table.add_row(
                item["question_id"], f"{item['awarded']:g}/{item['max_marks']:g}", escape(item["reason"] or "")
            )
        console.print(table)
        raw = typer.prompt("Override marks? e.g. Q7a=1.5, Q8=3 (Enter to keep)", default="", show_default=False)
        overrides = {}
        for part in filter(None, (p.strip() for p in raw.split(","))):
            qid, _, marks = part.partition("=")
            overrides[qid.strip()] = float(marks)
        approved = typer.confirm("Approve this student's marks?", default=True)
        return {"overrides": overrides, "approved": approved}
    table.add_column("Confidence", justify="right")
    table.add_column("What the model read")
    for item in payload["items"]:
        table.add_row(item["question_id"], f"{item['confidence']:.2f}", escape(item["text"][:200] or "(blank)"))
    console.print(table)
    corrections = {}
    for item in payload["items"]:
        fixed = typer.prompt(f"Correct {item['question_id']}? (Enter to keep)", default="", show_default=False)
        if fixed:
            corrections[item["question_id"]] = fixed
    return {"approved": True, "corrections": corrections}


def main(
    students: Annotated[int, typer.Option(help="How many students to grade")] = 3,
    from_transcripts: Annotated[
        bool, typer.Option("--from-transcripts", help="Skip vision: use transcripts.json as the answers")
    ] = False,
    auto_approve: Annotated[bool, typer.Option("--auto-approve", help="Approve every review automatically")] = False,
    thread: Annotated[str | None, typer.Option(help="Thread ID; re-use one to resume a paused batch")] = None,
    exam_id: Annotated[str, typer.Option("--exam", help="Exam ID")] = "cs_midterm",
    feedback: Annotated[str, typer.Option(help="'llm' or 'rules' (free) for strengths/improvements")] = "llm",
    runs: Annotated[int, typer.Option(help="Self-consistency runs per answer (💰 n× the cost)")] = 1,
    fake: Annotated[bool, typer.Option("--fake", help="Free offline run with the fake keyword grader")] = False,
) -> None:
    banner("Week 5 · The GradeWise orchestrator", "answer sheets → grades → human review → report cards")
    thread = thread or f"batch-{datetime.now():%Y%m%d-%H%M%S}"
    try:
        exam = resolve_exam(exam_id)
        if from_transcripts or fake:
            items: list[Any] = sheets_from_transcripts(exam, students=students)
            step(f"{len(items)} answer sheet(s) built from transcripts.json (no vision calls)")
        else:
            pdfs = sorted(SHEETS_DIR.glob("student_*.pdf"))[:students]
            if not pdfs:
                raise DataNotFoundError(f"No answer-sheet PDFs in {SHEETS_DIR}")
            items = pdfs
            step(f"{len(items)} handwritten PDF(s) to transcribe from {SHEETS_DIR.relative_to(REPO_ROOT)}")

        build: dict[str, Any] = {"feedback": "rules" if fake else feedback, "consistency_runs": runs}
        if fake:
            from langchain_core.embeddings import DeterministicFakeEmbedding

            from gradewise.grading.fakes import KeywordGraderModel
            from gradewise.orchestrator import sqlite_checkpointer
            from gradewise.rag import get_retriever

            warn("--fake: a keyword-matching stand-in is grading. The marks mean nothing; the plumbing is real.")
            fake_embeddings = DeterministicFakeEmbedding(size=32)
            build.update(
                llm=KeywordGraderModel(),
                retriever=get_retriever(mode="bm25", k=4, embeddings=fake_embeddings),
                checkpointer=sqlite_checkpointer(FAKE_DIR / "checkpoints.db"),
                db=FAKE_DIR / "gradewise.db",
                embeddings=fake_embeddings,
                persist_dir=FAKE_DIR / "vector_db",
                reports_dir=FAKE_DIR / "reports",
            )
        step(f"Thread [accent]{thread}[/] (re-run with --thread {thread} to resume if you stop half-way)")
        result = grade_batch(
            exam_id,
            items,
            thread_id=thread,
            auto_approve=auto_approve,
            on_event=on_event,
            reviewer=None if auto_approve else ask_teacher,
            exam=exam,
            **build,
        )
    except GradeWiseError as err:
        err.show()
        raise typer.Exit(code=1) from None

    if result.status == "paused":
        tip(f"Paused for review. Resume any time with: --thread {thread}")
        return
    table = Table(title="📊 Report cards")
    for col in ("Student", "Total", "%", "Band", "Review", "Report card"):
        table.add_column(col, style="accent" if col == "Student" else None)
    for r in result.reports:
        html = Path(result.files[r.student_id]["html"])
        table.add_row(
            r.student_id,
            f"{r.total:g}/{r.max_total:g}",
            f"{r.percentage:g}",
            r.grade_band,
            "⚑" if r.needs_review else "✓",
            str(html.relative_to(REPO_ROOT)) if html.is_relative_to(REPO_ROOT) else str(html),
        )
    console.print(table)
    success(
        "Batch graded",
        {"Thread": thread, "Students": len(result.reports)},
        next_step="Open a report card in your browser, then ask about the class: uv run week5/project/report_qa.py",
    )


if __name__ == "__main__":
    typer.run(main)
