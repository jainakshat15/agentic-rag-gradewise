"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧱 Week 4 · Day 5: Transcription demo (milestone M4)                │
│                                                                      │
│  What: reads ONE handwritten answer sheet with the transcription     │
│        subgraph and prints what it found, question by question.      │
│        If ground truth exists, it also prints the CER per answer.    │
│  Run:  uv run week4/project/transcription_graph.py                   │
│        uv run week4/project/transcription_graph.py PDF --dpi 150     │
│        (PDF e.g. data/answer-sheets/synthetic/student_03.pdf)        │
│  ⏮ Built on: week4/day3-day5 notebooks                              │
│  ⏭ Used by: week5 (grading reads these AnswerSheets)                │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from gradewise.config import DATA_DIR, OUTPUT_DIR, REPO_ROOT
from gradewise.console import banner, console, tip, warn
from gradewise.errors import DataNotFoundError, GradeWiseError
from gradewise.schemas.exam import ExamPaper
from gradewise.transcription import cer, needs_review, transcribe_sheet

DEFAULT_SHEET = DATA_DIR / "answer-sheets" / "synthetic" / "student_03.pdf"
DEFAULT_EXAM = DATA_DIR / "exams" / "cs_midterm" / "expected_exam.json"


def load_exam(path: Path) -> ExamPaper:
    if not path.exists():
        raise DataNotFoundError(f"Exam JSON not found: {path}")
    return ExamPaper.model_validate_json(path.read_text(encoding="utf-8"))


def load_ground_truth(pdf: Path) -> dict[str, str] | None:
    """The synthetic sheets ship with what we MEANT the student to write: transcripts.json."""
    truth_file = pdf.parent / "transcripts.json"
    if not truth_file.exists():
        return None
    return json.loads(truth_file.read_text(encoding="utf-8")).get(pdf.stem)


def main(
    pdf: Annotated[Path, typer.Argument(help="The handwritten answer-sheet PDF")] = DEFAULT_SHEET,
    exam_path: Annotated[Path, typer.Option("--exam", help="The ExamPaper JSON to map answers to")] = DEFAULT_EXAM,
    dpi: Annotated[int, typer.Option(help="Render resolution: try 150 vs 200 vs 300, compare CER and cost")] = 200,
    no_verify: Annotated[bool, typer.Option("--no-verify", help="Skip the double-pass check")] = False,
) -> None:
    banner("Week 4 · Transcription demo", "Handwritten PDF → verbatim answers, mapped to question IDs")
    pdf = pdf if pdf.is_absolute() or pdf.exists() else REPO_ROOT / pdf
    try:
        exam = load_exam(exam_path)
        sheet = transcribe_sheet(pdf, exam, pdf.stem, dpi=dpi, verify=not no_verify)
    except GradeWiseError as err:
        # 💡 e.g. MissingAPIKeyError: a friendly red panel instead of a stack trace
        err.show()
        raise typer.Exit(code=1) from None

    truth = load_ground_truth(pdf)
    table = Table(title=f"📝 {sheet.student_id}: {sheet.page_count} pages", show_lines=False)
    table.add_column("Question", style="accent", no_wrap=True)
    table.add_column("Pages", justify="center")
    table.add_column("Confidence", justify="right")
    if truth:
        table.add_column("CER", justify="right")
    table.add_column("Transcription (first 60 characters)")

    for answer in sheet.answers:
        conf_style = "err" if needs_review(answer) else "ok"
        preview = "[muted](blank)[/]" if answer.is_blank else escape(answer.text.replace("\n", " ⏎ ")[:60])
        row = [
            answer.question_id,
            ",".join(map(str, answer.pages)) or "-",
            f"[{conf_style}]{answer.confidence:.2f}[/]",
        ]
        if truth:
            reference = truth.get(answer.question_id)
            row.append("-" if reference is None else f"{cer(reference, answer.text):.1%}")
        row.append(preview)
        table.add_row(*row)
    console.print(table)

    if sheet.unmapped_text:
        warn(f"{len(sheet.unmapped_text)} snippet(s) could not be linked to a question:")
        for snippet in sheet.unmapped_text:
            console.print(f"   [muted]{escape(snippet[:100])}[/]")
    if truth:
        scored = [a for a in sheet.answers if a.question_id in truth]
        mean = sum(cer(truth[a.question_id], a.text) for a in scored) / max(1, len(scored))
        tip(f"Mean CER against ground truth: {mean:.1%} (lower is better; week 5 tracks this as a regression test)")

    out = OUTPUT_DIR / "transcriptions" / f"{sheet.student_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sheet.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[muted]💾 Saved {out.relative_to(REPO_ROOT)}[/]")


if __name__ == "__main__":
    typer.run(main)
