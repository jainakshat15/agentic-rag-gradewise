"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 5 · Day 5: How well did we read the handwriting?            │
│                                                                      │
│  What this file does                                                 │
│    Compares what the vision model read (AnswerSheets in              │
│    output/transcriptions/) with what the students REALLY wrote       │
│    (data/answer-sheets/synthetic/transcripts.json):                  │
│                                                                      │
│      CER → character error rate: catches "semafore" vs "semaphore"   │
│      WER → word error rate: easier to explain to Asha                │
│                                                                      │
│    per student, per question, and overall. A grade can only be as    │
│    good as the transcription it was based on, so this runs first.    │
│                                                                      │
│  Run : uv run week5/evaluation/run_evals.py --only transcription     │
│  ⏮ Built on: gradewise/transcription (cer, wer)                     │
│  ⏭ Used by: run_all(), tests/test_evals.py                          │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from gradewise.config import DATA_DIR, OUTPUT_DIR
from gradewise.errors import DataNotFoundError
from gradewise.evals.compat import import_pandas
from gradewise.evals.grading_eval import write_json
from gradewise.schemas.answers import AnswerSheet
from gradewise.transcription.confidence import cer, wer

pd = import_pandas()  # 💡 = `import pandas as pd`, with a fix for locked-down Windows

TRANSCRIPTS = DATA_DIR / "answer-sheets" / "synthetic" / "transcripts.json"
TRANSCRIPTIONS_DIR = OUTPUT_DIR / "transcriptions"


def load_ground_truth(path: str | Path = TRANSCRIPTS) -> dict[str, dict[str, str]]:
    """{"student_01": {"Q1": "what they wrote", ...}}. A question missing for a student = left blank."""
    path = Path(path)
    if not path.exists():
        raise DataNotFoundError(f"Ground-truth transcripts not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_sheets(folder: str | Path = TRANSCRIPTIONS_DIR) -> dict[str, AnswerSheet]:
    """Every AnswerSheet JSON in `folder` (written by `gradewise transcribe` or week4/project)."""
    folder = Path(folder)
    sheets = {}
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        sheet = AnswerSheet.model_validate_json(path.read_text(encoding="utf-8"))
        sheets[sheet.student_id] = sheet
    return sheets


def evaluate_transcriptions(
    sheets: Mapping[str, AnswerSheet] | Iterable[AnswerSheet],
    truth: Mapping[str, Mapping[str, str]],
    *,
    lowercase: bool = False,
) -> dict[str, Any]:
    """CER/WER for every (student, question) that has both a transcription and a ground truth.

    💡 The question list comes from the ANSWER SHEET (every exam question), so a question the student
       left blank counts too: reading text that isn't there is an error as well.
    """
    if not isinstance(sheets, Mapping):
        sheets = {s.student_id: s for s in sheets}
    rows = []
    for sid, sheet in sheets.items():
        if sid not in truth:
            continue
        for answer in sheet.answers:
            reference = truth[sid].get(answer.question_id, "")
            rows.append(
                {
                    "student_id": sid,
                    "question_id": answer.question_id,
                    "cer": round(cer(reference, answer.text, lowercase=lowercase), 4),
                    "wer": round(wer(reference, answer.text, lowercase=lowercase), 4),
                    "confidence": answer.confidence,
                    "chars": len(reference),
                }
            )
    if not rows:
        return {
            "rows": [],
            "per_student": [],
            "per_question": [],
            "overall": {"n": 0, "cer": math.nan, "wer": math.nan},
        }
    df = pd.DataFrame(rows)
    # 💡 Overall CER is weighted by length: a 5-character slip in a 2-word answer shouldn't count as much
    #    as the same mean over a 200-word essay. We report the simple mean too.
    weighted = float((df.cer * df.chars).sum() / max(1, df.chars.sum()))
    per_student = df.groupby("student_id").agg(cer=("cer", "mean"), wer=("wer", "mean"), answers=("cer", "size"))
    per_question = df.groupby("question_id", sort=False).agg(cer=("cer", "mean"), wer=("wer", "mean"))
    # ⚠️ The confidence check: do low-confidence answers really have higher error? If not, the
    #    review threshold is flagging the wrong answers.
    low = df[df.confidence < 0.75]
    return {
        "overall": {
            "n": len(df),
            "cer": round(float(df.cer.mean()), 4),
            "wer": round(float(df.wer.mean()), 4),
            "cer_weighted": round(weighted, 4),
            "low_confidence_cer": round(float(low.cer.mean()), 4) if len(low) else None,
            "high_confidence_cer": round(float(df[df.confidence >= 0.75].cer.mean()), 4)
            if len(df) > len(low)
            else None,
        },
        "per_student": per_student.round(4).reset_index().to_dict("records"),
        "per_question": per_question.round(4).reset_index().to_dict("records"),
        "rows": rows,
    }


def print_transcription_eval(result: dict[str, Any]) -> None:
    from rich.table import Table

    from gradewise.console import console

    table = Table(title="✍️ Transcription vs ground truth, per student")
    for col in ("Student", "Answers", "Mean CER", "Mean WER"):
        table.add_column(
            col, justify="left" if col == "Student" else "right", style="accent" if col == "Student" else None
        )
    for s in result["per_student"]:
        style = "err" if s["cer"] > 0.15 else ("warn" if s["cer"] > 0.05 else "ok")
        table.add_row(s["student_id"], str(s["answers"]), f"[{style}]{s['cer']:.1%}[/]", f"{s['wer']:.1%}")
    o = result["overall"]
    table.add_section()
    table.add_row("ALL", str(o["n"]), f"{o['cer']:.1%}", f"{o['wer']:.1%}")
    console.print(table)
    worst = sorted(result["per_question"], key=lambda q: q["cer"], reverse=True)[:3]
    if worst:
        console.print(
            "[muted]Hardest questions to read:[/] " + ", ".join(f"{q['question_id']} ({q['cer']:.1%})" for q in worst)
        )
    if o.get("low_confidence_cer") is not None and o.get("high_confidence_cer") is not None:
        console.print(
            f"[muted]CER when confidence < 0.75:[/] {o['low_confidence_cer']:.1%}  vs  "
            f"[muted]≥ 0.75:[/] {o['high_confidence_cer']:.1%}  (the first should be higher!)"
        )


def run_transcription_eval(
    sheets: Mapping[str, AnswerSheet] | Iterable[AnswerSheet] | None = None,
    *,
    truth_path: str | Path = TRANSCRIPTS,
    folder: str | Path = TRANSCRIPTIONS_DIR,
    show: bool = True,
    save: bool = True,
) -> dict[str, Any]:
    """Evaluate `sheets` (default: every AnswerSheet JSON in output/transcriptions/) against transcripts.json."""
    truth = load_ground_truth(truth_path)
    sheets = load_sheets(folder) if sheets is None else sheets
    if not sheets:
        raise DataNotFoundError(
            f"No transcribed answer sheets found in {folder}.",
            cause="Nothing has been transcribed yet.",
            fix="Run `uv run week4/project/transcription_graph.py data/answer-sheets/synthetic/student_01.pdf` "
            "(needs a vision model key), or `uv run python -m gradewise transcribe <pdf>`.",
        )
    result = evaluate_transcriptions(sheets, truth)
    result["truth_file"] = Path(truth_path).name
    if save:
        result["saved_to"] = str(write_json(result, "transcription_eval.json"))
    if show:
        print_transcription_eval(result)
    return result
