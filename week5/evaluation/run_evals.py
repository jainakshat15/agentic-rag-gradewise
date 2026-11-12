"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧱 Week 5 · Day 5: Is GradeWise fair? Grader vs human               │
│                                                                      │
│  What: compares GradeWise's marks with the golden human labels       │
│        (MAE, exact %, within-1 %, quadratic weighted kappa) and its  │
│        transcriptions with the true text (CER, WER).                 │
│  Run:  uv run week5/evaluation/run_evals.py                          │
│          → evaluates the reports already stored for cs_midterm       │
│        uv run week5/evaluation/run_evals.py --grade                  │
│          → first grades the synthetic students from transcripts.json │
│            with your grading model (💰 ~150 answers, well under $1)  │
│        uv run week5/evaluation/run_evals.py --fake                   │
│          → the same, free, with the offline keyword grader           │
│        uv run week5/evaluation/run_evals.py --only transcription     │
│  📦 DATA_SIZE=small: 2 students × Q1–Q3 (8 answers): only a check    │
│     that it runs, the metrics are far too noisy to trust             │
│  Output: rich tables + output/evals/grading_eval.json, ...           │
│  ⏮ Built on: gradewise/evals   ⏭ Used by: week5/day5 notebook       │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gradewise import datasets  # noqa: E402
from gradewise.config import OUTPUT_DIR  # noqa: E402
from gradewise.console import banner, step, success, tip, warn  # noqa: E402
from gradewise.errors import GradeWiseError  # noqa: E402
from gradewise.evals import run_grading_eval, run_transcription_eval  # noqa: E402
from gradewise.evals.grading_eval import GOLDEN_CSV  # noqa: E402
from gradewise.orchestrator import grade_batch, resolve_exam, sheets_from_transcripts  # noqa: E402

FAKE_DIR = OUTPUT_DIR / "fake_run"


def grade_synthetic(exam_id: str, *, fake: bool) -> list:
    """Grade the synthetic students from the ground-truth transcripts (so only the GRADER is tested).

    📦 DATA_SIZE decides who and what: 10 students × every question, or 2 students × Q1–Q3 when small.
    """
    exam = datasets.subset_exam(resolve_exam(exam_id, quiet=True))
    sheets = sheets_from_transcripts(exam, students=datasets.synthetic_students())
    build: dict[str, Any] = {"feedback": "rules", "pdf": False}
    if fake:
        from langchain_core.embeddings import DeterministicFakeEmbedding

        from gradewise.grading.fakes import KeywordGraderModel
        from gradewise.rag import get_retriever

        emb = DeterministicFakeEmbedding(size=32)
        build.update(
            checkpointer="memory",
            llm=KeywordGraderModel(),
            retriever=get_retriever(mode="bm25", k=4, embeddings=emb),
            db=FAKE_DIR / "gradewise.db",
            embeddings=emb,
            persist_dir=FAKE_DIR / "vector_db",
            reports_dir=FAKE_DIR / "reports",
        )
    step(f"Grading {len(sheets)} students from transcripts.json ({'fake keyword grader' if fake else 'your model'})…")
    # 💡 auto_approve: for an evaluation we want the grader's OWN marks, untouched by a reviewer.
    result = grade_batch(
        exam_id, sheets, thread_id=f"eval-{datetime.now():%Y%m%d-%H%M%S}", auto_approve=True, exam=exam, **build
    )
    return result.reports


def main(
    only: Annotated[str, typer.Option(help="'all', 'grading' or 'transcription'")] = "all",
    grade: Annotated[
        bool, typer.Option("--grade", help="Grade the synthetic students first (uses your model)")
    ] = False,
    fake: Annotated[bool, typer.Option("--fake", help="Grade first with the free offline fake grader")] = False,
    exam_id: Annotated[str, typer.Option("--exam", help="Exam ID")] = "cs_midterm",
    golden: Annotated[Path, typer.Option(help="Golden labels CSV")] = GOLDEN_CSV,
) -> None:
    banner("Week 5 · Evaluation", "GradeWise vs the human examiner")
    step(datasets.describe())
    if datasets.is_small():
        warn("DATA_SIZE=small: these metrics rest on a handful of answers. Use them to check it runs, not to judge.")
    summary: dict[str, Any] = {}
    try:
        if only in ("all", "transcription"):
            try:
                t = run_transcription_eval()
                summary["Mean CER"] = f"{t['overall']['cer']:.1%}"
                summary["Mean WER"] = f"{t['overall']['wer']:.1%}"
            except GradeWiseError as err:
                warn(f"Transcription eval skipped: {err.what}")
                tip("Transcribe some sheets first (week4/project/transcription_graph.py), then re-run.")
        if only in ("all", "grading"):
            reports = grade_synthetic(exam_id, fake=fake) if (grade or fake) else None
            g = run_grading_eval(exam_id, reports=reports, golden_path=golden)
            summary.update(
                {
                    "MAE (marks)": f"{g['overall']['mae']:.2f}",
                    "Exact agreement": f"{g['overall']['exact']:.0%}",
                    "Within 1 mark": f"{g['overall']['within_1']:.0%}",
                    "Quadratic weighted kappa": f"{g['overall']['qwk']:.2f}",
                }
            )
            if fake:
                warn("These are the FAKE grader's numbers: a baseline to beat, not GradeWise's real quality.")
    except GradeWiseError as err:
        err.show()
        raise typer.Exit(code=1) from None
    if summary:
        success(
            "Evaluation finished",
            {**summary, "Saved to": "output/evals/"},
            next_step="QWK ≥ 0.8 is strong agreement. Find the worst question above and read its justifications.",
        )


if __name__ == "__main__":
    typer.run(main)
