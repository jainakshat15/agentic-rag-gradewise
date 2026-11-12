"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 5 · Day 5: Grader vs human                                  │
│                                                                      │
│  What this file does                                                 │
│    Compares GradeWise's marks with a human examiner's (the golden    │
│    labels), question by question:                                    │
│                                                                      │
│      MAE            → on average, how many marks apart are we?       │
│      exact %        → how often do we give the very same mark?       │
│      within-1 %     → how often are we at most one mark apart?       │
│      QWK            → quadratic weighted kappa: agreement beyond     │
│                       chance, where big disagreements cost more      │
│      confusion      → do we mark too HARSHLY or too GENEROUSLY?      │
│                                                                      │
│  Run : uv run week5/evaluation/run_evals.py                          │
│  ⏮ Built on: data/golden/*.csv, stored ReportCards (gradewise/db.py) │
│  ⏭ Used by: run_all() in this package, tests/test_evals.py          │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gradewise.config import DATA_DIR, OUTPUT_DIR
from gradewise.errors import DataNotFoundError, GradeWiseError
from gradewise.evals.compat import import_pandas
from gradewise.schemas.report import ReportCard

pd = import_pandas()  # 💡 = `import pandas as pd`, with a fix for locked-down Windows

GOLDEN_CSV = DATA_DIR / "golden" / "synthetic_labels.csv"
EVAL_DIR = OUTPUT_DIR / "evals"

Key = tuple[str, str]  # (student_id, question_id)


class NoPredictionsError(GradeWiseError):
    title = "Nothing to evaluate yet"
    cause = "No stored report cards match the students in the golden labels."
    fix = "Grade the synthetic sheets first: `uv run week5/project/orchestrator.py --from-transcripts --auto-approve`."
    help_ref = "week5/evaluation/README.md"


# ── 📥 Loading ─────────────────────────────────────────────────────────────────
def load_golden(path: str | Path = GOLDEN_CSV) -> pd.DataFrame:
    """The human marks: one row per (student, question, criterion) with `awarded` and `max`."""
    path = Path(path)
    if not path.exists():
        raise DataNotFoundError(f"Golden labels not found: {path}")
    df = pd.read_csv(path, dtype={"student_id": str, "question_id": str, "criterion_id": str})
    missing = {"student_id", "question_id", "criterion_id", "awarded", "max"} - set(df.columns)
    if missing:
        raise DataNotFoundError(f"{path.name} is missing columns: {sorted(missing)}")
    return df.dropna(subset=["awarded"])


def human_marks(golden: pd.DataFrame) -> tuple[dict[Key, float], dict[str, float]]:
    """Sum the criterion marks → ({(student, question): marks}, {question: max marks})."""
    per_q = golden.groupby(["student_id", "question_id"], sort=False)["awarded"].sum()
    max_marks = golden.groupby(["student_id", "question_id"], sort=False)["max"].sum().groupby("question_id").max()
    return {k: float(v) for k, v in per_q.items()}, {k: float(v) for k, v in max_marks.items()}


def predicted_marks(reports: Iterable[ReportCard]) -> dict[Key, float]:
    """{(student, question): GradeWise's mark} from ReportCards."""
    return {(rep.student_id, r.question_id): r.awarded_marks for rep in reports for r in rep.results}


# ── 📏 The metrics (pure functions, tested on toy data) ──────────────────────────
def mae(human: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute error in marks. 0 is perfect."""
    h, p = np.asarray(human, float), np.asarray(predicted, float)
    return float(np.mean(np.abs(h - p))) if len(h) else math.nan


def exact_agreement(human: Sequence[float], predicted: Sequence[float]) -> float:
    """Share of answers given exactly the same mark (half marks count)."""
    h, p = np.asarray(human, float), np.asarray(predicted, float)
    return float(np.mean(np.isclose(h, p))) if len(h) else math.nan


def within(human: Sequence[float], predicted: Sequence[float], tolerance: float = 1.0) -> float:
    """Share of answers at most `tolerance` marks apart."""
    h, p = np.asarray(human, float), np.asarray(predicted, float)
    return float(np.mean(np.abs(h - p) <= tolerance + 1e-9)) if len(h) else math.nan


def round_marks(marks: Sequence[float]) -> list[int]:
    """Round half UP (2.5 → 3). ⚠️ Python's round() rounds half to EVEN (2.5 → 2), which would bias kappa."""
    return [int(math.floor(m + 0.5)) for m in marks]


def quadratic_weighted_kappa(human: Sequence[float], predicted: Sequence[float]) -> float:
    """Cohen's kappa with quadratic weights, on marks rounded to whole numbers.

    💡 1.0 = perfect agreement, 0 = no better than chance, < 0 = worse than chance. "Quadratic" means
       being 3 marks apart costs 9× as much as being 1 apart, which is how a moderator thinks.
    ⚠️ Kappa is undefined when both raters give ONE single mark to everybody (no variation to agree on).
       We return 1.0 if they agree exactly in that case, and 0.0 otherwise.
    """
    from sklearn.metrics import cohen_kappa_score

    h, p = round_marks(human), round_marks(predicted)
    if not h:
        return math.nan
    if len(set(h) | set(p)) == 1:
        return 1.0
    labels = list(range(min(h + p), max(h + p) + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = cohen_kappa_score(h, p, weights="quadratic", labels=labels)
    if math.isnan(kappa):
        return 1.0 if h == p else 0.0
    return float(kappa)


def confusion_summary(human: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    """Is the grader too generous or too harsh? Counts plus a human × grader matrix of rounded marks."""
    h, p = np.asarray(human, float), np.asarray(predicted, float)
    diff = p - h
    hr, pr = round_marks(h), round_marks(p)
    labels = sorted(set(hr) | set(pr))
    matrix = pd.crosstab(pd.Series(hr, name="human"), pd.Series(pr, name="grader")).reindex(
        index=labels, columns=labels, fill_value=0
    )
    return {
        "agree": int(np.sum(np.isclose(diff, 0))),
        "grader_higher": int(np.sum(diff > 1e-9)),
        "grader_lower": int(np.sum(diff < -1e-9)),
        "mean_signed_error": float(np.mean(diff)) if len(diff) else math.nan,
        "matrix": {str(r): {str(c): int(matrix.loc[r, c]) for c in labels} for r in labels},
    }


def _metrics(human: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    return {
        "n": len(human),
        "mae": round(mae(human, predicted), 3),
        "exact": round(exact_agreement(human, predicted), 3),
        "within_1": round(within(human, predicted, 1.0), 3),
        "qwk": round(quadratic_weighted_kappa(human, predicted), 3),
    }


def compare_marks(
    human: Mapping[Key, float], predicted: Mapping[Key, float], max_marks: Mapping[str, float] | None = None
) -> dict[str, Any]:
    """Every metric, overall and per question, over the (student, question) pairs both sides have."""
    keys = [k for k in human if k in predicted]
    if not keys:
        raise NoPredictionsError("No (student, question) pair has both a human mark and a GradeWise mark.")
    h = [human[k] for k in keys]
    p = [predicted[k] for k in keys]
    per_question = []
    for qid in dict.fromkeys(k[1] for k in keys):
        qk = [k for k in keys if k[1] == qid]
        qh, qp = [human[k] for k in qk], [predicted[k] for k in qk]
        per_question.append(
            {
                "question_id": qid,
                "max_marks": (max_marks or {}).get(qid),
                "human_mean": round(float(np.mean(qh)), 2),
                "grader_mean": round(float(np.mean(qp)), 2),
                **_metrics(qh, qp),
            }
        )
    gaps = sorted(keys, key=lambda k: abs(predicted[k] - human[k]), reverse=True)
    return {
        "overall": _metrics(h, p),
        "per_question": per_question,
        "confusion": confusion_summary(h, p),
        "largest_gaps": [
            {"student_id": s, "question_id": q, "human": human[(s, q)], "grader": predicted[(s, q)]}
            for s, q in gaps[:8]
            if not math.isclose(human[(s, q)], predicted[(s, q)])
        ],
        "students": sorted({k[0] for k in keys}),
    }


def criterion_agreement(golden: pd.DataFrame, reports: Iterable[ReportCard]) -> float:
    """Share of individual criteria where GradeWise gave exactly the human's mark (stricter than per question)."""
    ours = {(rep.student_id, c.criterion_id): c.awarded for rep in reports for r in rep.results for c in r.criteria}
    pairs = [
        (row.awarded, ours[(row.student_id, row.criterion_id)])
        for row in golden.itertuples()
        if (row.student_id, row.criterion_id) in ours
    ]
    return round(float(np.mean([math.isclose(a, b) for a, b in pairs])), 3) if pairs else math.nan


# ── 🖨️ Reporting ───────────────────────────────────────────────────────────────
def print_grading_eval(result: dict[str, Any]) -> None:
    """Rich tables: per question, overall, and the confusion summary."""
    from rich.table import Table

    from gradewise.console import console

    table = Table(title="🎯 GradeWise vs the human examiner, per question")
    for col in ("Question", "Max", "Human mean", "Grader mean", "MAE", "Exact", "Within 1", "QWK"):
        table.add_column(
            col, justify="right" if col != "Question" else "left", style="accent" if col == "Question" else None
        )
    for q in result["per_question"]:
        style = "err" if q["mae"] >= 1 else ("warn" if q["mae"] >= 0.5 else "ok")
        table.add_row(
            q["question_id"],
            f"{q['max_marks']:g}" if q["max_marks"] else "-",
            f"{q['human_mean']:g}",
            f"{q['grader_mean']:g}",
            f"[{style}]{q['mae']:.2f}[/]",
            f"{q['exact']:.0%}",
            f"{q['within_1']:.0%}",
            f"{q['qwk']:.2f}",
        )
    o = result["overall"]
    table.add_section()
    table.add_row("ALL", "", "", "", f"{o['mae']:.2f}", f"{o['exact']:.0%}", f"{o['within_1']:.0%}", f"{o['qwk']:.2f}")
    console.print(table)
    c = result["confusion"]
    console.print(
        f"[muted]Answers compared:[/] {o['n']}   [ok]same mark {c['agree']}[/]   "
        f"[warn]grader higher {c['grader_higher']}[/]   [err]grader lower {c['grader_lower']}[/]   "
        f"[muted]mean signed error {c['mean_signed_error']:+.2f}[/]"
    )
    if "criterion_exact" in result:
        console.print(f"[muted]Criterion-level exact agreement:[/] {result['criterion_exact']:.0%}")


def write_json(result: dict[str, Any], name: str) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / name
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return path


def evaluate_grading(
    reports: Sequence[ReportCard],
    golden_path: str | Path = GOLDEN_CSV,
    *,
    show: bool = True,
    save: bool = True,
) -> dict[str, Any]:
    """Compare these ReportCards with the golden labels; print and save (output/evals/grading_eval.json).

    📦 DATA_SIZE=small compares only the small slice: student_01 + student_05 × Q1–Q3 (datasets.golden_rows).
    """
    from gradewise import datasets

    golden = datasets.golden_rows(load_golden(golden_path))
    human, max_marks = human_marks(golden)
    result = compare_marks(human, predicted_marks(reports), max_marks)
    result["criterion_exact"] = criterion_agreement(golden, reports)
    result["golden_file"] = Path(golden_path).name
    if save:
        result["saved_to"] = str(write_json(result, "grading_eval.json"))
    if show:
        print_grading_eval(result)
    return result


def run_grading_eval(
    exam_id: str = "cs_midterm",
    *,
    reports: Sequence[ReportCard] | None = None,
    golden_path: str | Path = GOLDEN_CSV,
    db: Any = None,
    show: bool = True,
    save: bool = True,
) -> dict[str, Any]:
    """Evaluate the reports stored for `exam_id` (or the `reports` you pass) against the golden labels."""
    if reports is None:
        from gradewise.db import list_reports

        reports = list_reports(exam_id, db=db)
    if not reports:
        raise NoPredictionsError(f'No report cards are stored for exam "{exam_id}".')
    return evaluate_grading(reports, golden_path, show=show, save=save)
