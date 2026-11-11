"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧱 Week 5 · Day 5: gradewise.evals, is GradeWise any good?          │
│                                                                      │
│  What: numbers that tell Asha whether to trust the marks.            │
│      grading_eval.py       → MAE, exact %, within-1 %, QWK           │
│      transcription_eval.py → CER / WER per student and question      │
│      run_all()             → both, printed and saved to output/evals │
│  Run : uv run week5/evaluation/run_evals.py                          │
│  ⏮ Built on: data/golden, transcripts.json, stored reports          │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from typing import Any

from gradewise.console import warn
from gradewise.errors import GradeWiseError
from gradewise.evals.grading_eval import (
    compare_marks,
    confusion_summary,
    evaluate_grading,
    exact_agreement,
    mae,
    quadratic_weighted_kappa,
    run_grading_eval,
    within,
)
from gradewise.evals.transcription_eval import evaluate_transcriptions, run_transcription_eval

__all__ = [
    "compare_marks",
    "confusion_summary",
    "evaluate_grading",
    "evaluate_transcriptions",
    "exact_agreement",
    "mae",
    "quadratic_weighted_kappa",
    "run_all",
    "run_grading_eval",
    "run_transcription_eval",
    "within",
]


def run_all(exam_id: str = "cs_midterm", *, show: bool = True, **kwargs: Any) -> dict[str, Any]:
    """Run every evaluation that has its inputs ready. Missing inputs are skipped with a warning.

    Returns {"transcription": result | None, "grading": result | None}.
    """
    results: dict[str, Any] = {"transcription": None, "grading": None}
    try:
        results["transcription"] = run_transcription_eval(show=show)
    except GradeWiseError as err:
        warn(f"Transcription eval skipped: {err.what}")
    try:
        results["grading"] = run_grading_eval(exam_id, show=show, **kwargs)
    except GradeWiseError as err:
        warn(f"Grading eval skipped: {err.what}")
    return results
