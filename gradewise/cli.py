"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧱 gradewise/cli.py: the `gradewise` command                        │
│                                                                      │
│      uv run gradewise --help      → list every command               │
│      uv run gradewise doctor      → run setup/diagnostics.py         │
│      uv run gradewise info        → show your current settings       │
│                                                                      │
│  New commands arrive as the course goes on:                          │
│      week 2 → `ingest` · week 3 → `exam` · week 5 → `grade`          │
│      week 6 → `serve`                                                │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Annotated

import typer

from gradewise import COURSE_NAME, __version__
from gradewise.config import REPO_ROOT, settings
from gradewise.console import banner, success, tip

app = typer.Typer(
    help=f"🎓 GradeWise: the capstone of '{COURSE_NAME}'.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def doctor() -> None:
    """🩺 Check that your machine is ready for the course."""
    sys.argv = ["diagnostics.py"]
    try:
        runpy.run_path(str(REPO_ROOT / "setup" / "diagnostics.py"), run_name="__main__")
    except SystemExit as exc:
        raise typer.Exit(code=int(exc.code or 0)) from None


@app.command()
def info() -> None:
    """⚙️ Show the settings GradeWise is using (keys are never printed)."""
    banner("GradeWise settings", f"version {__version__}")

    def key_status(secret) -> str:
        return "✅ set" if secret else "— not set"

    success(
        "Current configuration",
        {
            "Chat model": settings.chat_model,
            "Grading model": settings.grading_model,
            "Vision model": settings.transcription_model,
            "Embedding model": settings.embedding_model,
            "Budget mode": settings.budget_mode,
            "OpenAI key": key_status(settings.openai_api_key),
            "Anthropic key": key_status(settings.anthropic_api_key),
            "Google key": key_status(settings.google_api_key),
            "LangSmith tracing": settings.langsmith_tracing,
        },
        next_step="Change any of these in your .env file",
    )
    if not settings.budget_mode:
        tip("Set BUDGET_MODE=true in .env to run every step on the cheapest model.")


# ── 📚 Week 2: the knowledge base ──────────────────────────────────────────
@app.command()
def ingest(
    reset: bool = typer.Option(False, "--reset", help="Wipe the collection and rebuild it"),
    collection: str = typer.Option("cs201", help="Knowledge-base collection name"),
) -> None:
    """📚 Ingest data/knowledge-base/*.pdf into the vector store (safe to re-run)."""
    from gradewise.errors import GradeWiseError
    from gradewise.ingestion import ingest as run_ingest

    try:
        run_ingest(collection=collection, reset=reset)
    except GradeWiseError as err:
        err.show()
        raise typer.Exit(code=1) from None


# ── 📝 Week 3: exam understanding ──────────────────────────────────────────
exam_app = typer.Typer(help="📝 Exam understanding: exam pack → validated ExamPaper.", no_args_is_help=True)
app.add_typer(exam_app, name="exam")


@exam_app.command("extract")
def exam_extract(
    exam_id: str = typer.Argument("cs_midterm", help="Folder under data/exams/"),
    max_attempts: int = typer.Option(3, help="Extraction attempts before asking for review"),
    auto_approve: bool = typer.Option(False, "--auto-approve", help="Approve without asking"),
) -> None:
    """📝 Extract → validate → retry → approve → store an exam (week 3)."""
    from langgraph.types import Command

    from gradewise.errors import GradeWiseError
    from gradewise.exam.graph import build_exam_graph, print_exam_summary, start_input
    from gradewise.schemas import ExamPaper

    banner("Exam understanding", exam_id)
    try:
        graph = build_exam_graph(checkpointer="sqlite")
        config = {"configurable": {"thread_id": f"exam-{exam_id}"}}
        result = graph.invoke(start_input(exam_id, max_attempts=max_attempts), config)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            if payload["paper"]:
                print_exam_summary(ExamPaper.model_validate(payload["paper"]), payload["problems"])
            approved = auto_approve or typer.confirm("Approve this exam?", default=False)
            result = graph.invoke(Command(resume={"approved": approved}), config)
    except GradeWiseError as err:
        err.show()
        raise typer.Exit(code=1) from None
    if not result.get("stored"):
        tip("Not approved, so nothing was stored.")


# ── ✍️ Week 4: reading handwriting ─────────────────────────────────────────
@app.command()
def transcribe(
    pdf: Annotated[Path, typer.Argument(help="The handwritten answer-sheet PDF")],
    exam: Annotated[Path, typer.Option(help="The ExamPaper JSON to map answers to")] = (
        REPO_ROOT / "data" / "exams" / "cs_midterm" / "expected_exam.json"
    ),
    student: Annotated[str | None, typer.Option(help="Student ID (default: the PDF's file name)")] = None,
    dpi: Annotated[int, typer.Option(help="Render resolution (150-300)")] = 200,
    verify: Annotated[bool, typer.Option("--verify/--no-verify", help="Double-check low-confidence pages")] = True,
) -> None:
    """✍️ Transcribe a handwritten answer sheet into an AnswerSheet JSON (week 4)."""
    from gradewise.config import OUTPUT_DIR
    from gradewise.errors import DataNotFoundError, GradeWiseError
    from gradewise.schemas.exam import ExamPaper
    from gradewise.transcription import transcribe_sheet

    try:
        if not exam.exists():
            raise DataNotFoundError(f"Exam JSON not found: {exam}")
        paper = ExamPaper.model_validate_json(exam.read_text(encoding="utf-8"))
        sheet = transcribe_sheet(pdf, paper, student or pdf.stem, dpi=dpi, verify=verify)
    except GradeWiseError as err:
        err.show()
        raise typer.Exit(code=1) from None
    out = OUTPUT_DIR / "transcriptions" / f"{sheet.student_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sheet.model_dump_json(indent=2), encoding="utf-8")
    tip(f"Saved {out.relative_to(REPO_ROOT)}. Next: `uv run python -m gradewise grade` (week 5).")


# ── 🧮 Week 5: grading, reports, Q&A, evaluation ───────────────────────────
@app.command()
def grade(
    students: Annotated[int, typer.Option(help="How many synthetic students to grade")] = 3,
    exam_id: Annotated[str, typer.Option("--exam", help="Exam ID")] = "cs_midterm",
    from_transcripts: Annotated[
        bool, typer.Option("--from-transcripts", help="Skip vision: grade transcripts.json")
    ] = False,
    auto_approve: Annotated[bool, typer.Option("--auto-approve", help="Approve every review automatically")] = False,
    thread: Annotated[str | None, typer.Option(help="Thread ID; re-use one to resume a paused batch")] = None,
) -> None:
    """🧮 Grade a batch end to end: transcribe → grade → review → report cards (week 5)."""
    sys.argv = ["orchestrator.py", "--students", str(students), "--exam", exam_id]
    sys.argv += ["--from-transcripts"] * from_transcripts + ["--auto-approve"] * auto_approve
    sys.argv += ["--thread", thread] if thread else []
    try:
        runpy.run_path(str(REPO_ROOT / "week5" / "project" / "orchestrator.py"), run_name="__main__")
    except SystemExit as exc:
        raise typer.Exit(code=int(exc.code or 0)) from None


@app.command()
def report(
    student: Annotated[str, typer.Argument(help="Student ID, e.g. student_04")],
    exam_id: Annotated[str, typer.Option("--exam", help="Exam ID")] = "cs_midterm",
    pdf: Annotated[bool, typer.Option("--pdf/--no-pdf", help="Also write a PDF")] = True,
) -> None:
    """📄 Write a stored student's report card to output/reports/<exam>/ (HTML + PDF)."""
    from gradewise.db import get_report
    from gradewise.errors import GradeWiseError
    from gradewise.orchestrator import resolve_exam
    from gradewise.reports import save_report_files

    try:
        card = get_report(exam_id, student)
        paths = save_report_files(card, resolve_exam(exam_id, quiet=True), pdf=pdf)
    except GradeWiseError as err:
        err.show()
        raise typer.Exit(code=1) from None
    success(
        f"Report card for {student}",
        {"Total": f"{card.total:g}/{card.max_total:g} ({card.grade_band})", **{k.upper(): v for k, v in paths.items()}},
    )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help='e.g. "Which question did the class struggle with most?"')],
    exam_id: Annotated[str, typer.Option("--exam", help="Exam ID")] = "cs_midterm",
) -> None:
    """💬 Ask the 'chat with results' agent about an exam's report cards (week 5)."""
    from rich.markdown import Markdown

    from gradewise.console import console
    from gradewise.errors import GradeWiseError
    from gradewise.reports.qa_agent import ask as ask_results

    try:
        console.print(Markdown(ask_results(question, exam_id)))
    except GradeWiseError as err:
        err.show()
        raise typer.Exit(code=1) from None


@app.command()
def evals(
    only: Annotated[str, typer.Option(help="'all', 'grading' or 'transcription'")] = "all",
    grade_first: Annotated[bool, typer.Option("--grade", help="Grade the synthetic students first")] = False,
    fake: Annotated[bool, typer.Option("--fake", help="Grade first with the free offline fake grader")] = False,
) -> None:
    """📏 Grader vs human (MAE, QWK) and transcription CER/WER (week 5)."""
    sys.argv = ["run_evals.py", "--only", only] + ["--grade"] * grade_first + ["--fake"] * fake
    try:
        runpy.run_path(str(REPO_ROOT / "week5" / "evaluation" / "run_evals.py"), run_name="__main__")
    except SystemExit as exc:
        raise typer.Exit(code=int(exc.code or 0)) from None


# ── 🚢 Week 6: ship it ─────────────────────────────────────────────────────
@app.command()
def serve(
    what: Annotated[str, typer.Argument(help="'api' or 'ui'")] = "api",
    demo: Annotated[bool, typer.Option("--demo", help="Offline fakes, no API key")] = False,
    port: Annotated[int | None, typer.Option(help="Port (api 8000, ui 8501)")] = None,
    host: Annotated[str, typer.Option(help="0.0.0.0 to listen on every interface")] = "127.0.0.1",
) -> None:
    """🚢 Start the week 6 API (FastAPI) or dashboard (Streamlit)."""
    import subprocess

    if what == "api":
        from gradewise.api.__main__ import main as run_api

        run_api(["--host", host, "--port", str(port or 8000)] + ["--demo"] * demo)
    elif what == "ui":
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(REPO_ROOT / "gradewise" / "ui" / "app.py"),
            f"--server.port={port or 8501}",
            f"--server.address={host}",
        ]
        raise typer.Exit(subprocess.call(cmd))
    else:
        raise typer.BadParameter("Choose 'api' or 'ui'.")


if __name__ == "__main__":
    app()
