"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧪 tests/test_evals.py: the week 5 evaluation maths                 │
│                                                                      │
│  Run:  uv run python -m pytest tests/test_evals.py -q                │
│  💡 Toy numbers you can check by hand, then the real golden labels.  │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json

import pytest

from gradewise.config import DATA_DIR
from gradewise.evals import (
    compare_marks,
    confusion_summary,
    evaluate_grading,
    evaluate_transcriptions,
    exact_agreement,
    mae,
    quadratic_weighted_kappa,
    within,
)
from gradewise.evals.grading_eval import criterion_agreement, human_marks, load_golden, round_marks
from gradewise.evals.transcription_eval import load_ground_truth
from gradewise.schemas import AnswerSheet, CriterionResult, ExamPaper, GradeResult, ReportCard, TranscribedAnswer

GOLDEN = DATA_DIR / "golden" / "synthetic_labels.csv"
TRANSCRIPTS = DATA_DIR / "answer-sheets" / "synthetic" / "transcripts.json"
EXPECTED_EXAM = DATA_DIR / "exams" / "cs_midterm" / "expected_exam.json"
needs_golden = pytest.mark.skipif(not GOLDEN.exists(), reason="golden labels not generated yet")


# ── 📏 Toy maths ─────────────────────────────────────────────────────────────
def test_mae_exact_and_within_by_hand():
    human, grader = [2, 3, 0, 4], [2, 2, 1, 1.5]
    assert mae(human, grader) == pytest.approx((0 + 1 + 1 + 2.5) / 4)
    assert exact_agreement(human, grader) == 0.25
    assert within(human, grader, 1) == 0.75


def test_qwk_perfect_and_chance_and_reversed():
    assert quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert quadratic_weighted_kappa([0, 1, 2, 3], [3, 2, 1, 0]) == pytest.approx(-1.0)
    assert quadratic_weighted_kappa([2, 2, 2], [2, 2, 2]) == 1.0  # ⚠️ undefined in theory: we call it agreement


def test_qwk_matches_a_hand_computation():
    # Human [0,1,2], grader [0,2,2]. Weights w_ij = (i-j)²/(k-1)² with k=3 → off-by-one = 0.25.
    # Observed: one (1→2) disagreement → Σ w·O = 0.25. Expected (outer product of marginals / n):
    #   human marginals [1,1,1], grader [1,0,2] → E_ij = h_i·g_j/3; Σ w·E = (0.25·(0+... )) computed below.
    import numpy as np

    h_m, g_m = np.array([1, 1, 1]), np.array([1, 0, 2])
    w = np.array([[(i - j) ** 2 / 4 for j in range(3)] for i in range(3)])
    observed = np.zeros((3, 3))
    for i, j in [(0, 0), (1, 2), (2, 2)]:
        observed[i, j] += 1
    expected = np.outer(h_m, g_m) / 3
    kappa = 1 - (w * observed).sum() / (w * expected).sum()
    assert quadratic_weighted_kappa([0, 1, 2], [0, 2, 2]) == pytest.approx(kappa)


def test_marks_round_half_up():
    assert round_marks([0.5, 1.5, 2.5, 2.4]) == [1, 2, 3, 2]  # not Python's banker's rounding (2.5 → 2)


def test_confusion_summary():
    c = confusion_summary([1, 2, 3], [1, 3, 2.5])
    assert (c["agree"], c["grader_higher"], c["grader_lower"]) == (1, 1, 1)
    assert c["matrix"]["2"]["3"] == 1 and c["mean_signed_error"] == pytest.approx(0.5 / 3)


def test_compare_marks_per_question():
    human = {("a", "Q1"): 2, ("b", "Q1"): 1, ("a", "Q2"): 3, ("b", "Q2"): 0}
    grader = {("a", "Q1"): 2, ("b", "Q1"): 2, ("a", "Q2"): 3, ("b", "Q2"): 0, ("z", "Q9"): 5}
    result = compare_marks(human, grader, {"Q1": 2, "Q2": 3})
    q1, q2 = result["per_question"]
    assert q1["question_id"] == "Q1" and q1["mae"] == 0.5 and q1["exact"] == 0.5
    assert q2["mae"] == 0 and q2["qwk"] == pytest.approx(1.0)
    assert result["overall"]["n"] == 4  # the unmatched ("z", "Q9") pair is ignored
    assert result["largest_gaps"][0] == {"student_id": "b", "question_id": "Q1", "human": 1, "grader": 2}


# ── 📦 The real golden labels ────────────────────────────────────────────────
@needs_golden
def test_golden_labels_match_the_exam():
    exam = ExamPaper.model_validate_json(EXPECTED_EXAM.read_text(encoding="utf-8"))
    golden = load_golden(GOLDEN)
    criteria = {c.id: c.marks for q in exam.questions for c in q.criteria}
    assert set(golden.criterion_id) <= set(criteria)
    assert (golden.awarded <= golden["max"]).all() and (golden.awarded >= 0).all()
    _, max_marks = human_marks(golden)
    assert max_marks == {q.id: q.max_marks for q in exam.questions}


def _reports_from_golden(golden) -> list[ReportCard]:
    """A 'grader' that copies the human exactly, built from the golden CSV."""
    reports = []
    for sid, rows in golden.groupby("student_id"):
        results = []
        for qid, qrows in rows.groupby("question_id", sort=False):
            criteria = [
                CriterionResult(criterion_id=r.criterion_id, awarded=r.awarded, justification="")
                for r in qrows.itertuples()
            ]
            results.append(
                GradeResult(
                    question_id=qid,
                    awarded_marks=qrows.awarded.sum(),
                    max_marks=qrows["max"].sum(),
                    criteria=criteria,
                    feedback="",
                    confidence=1,
                )
            )
        reports.append(ReportCard.from_results(sid, "cs_midterm", results))
    return reports


@needs_golden
def test_a_perfect_grader_scores_perfectly():
    golden = load_golden(GOLDEN)
    reports = _reports_from_golden(golden)
    result = evaluate_grading(reports, GOLDEN, show=False, save=False)
    assert result["overall"]["mae"] == 0 and result["overall"]["exact"] == 1 and result["overall"]["qwk"] == 1
    assert result["criterion_exact"] == 1.0 and criterion_agreement(golden, reports) == 1.0


@needs_golden
def test_a_harsh_grader_is_caught():
    golden = load_golden(GOLDEN)
    harsh = [
        r.model_copy(
            update={
                "results": [g.model_copy(update={"awarded_marks": max(0.0, g.awarded_marks - 1)}) for g in r.results]
            }
        )
        for r in _reports_from_golden(golden)
    ]
    result = evaluate_grading(harsh, GOLDEN, show=False, save=False)
    assert result["confusion"]["grader_lower"] > result["confusion"]["grader_higher"] == 0
    assert result["overall"]["within_1"] == 1.0 and result["overall"]["mae"] > 0.5


# ── ✍️ Transcription ─────────────────────────────────────────────────────────
def _sheet(sid: str, texts: dict[str, str]) -> AnswerSheet:
    answers = [TranscribedAnswer(question_id=q, text=t, confidence=0.9) for q, t in texts.items()]
    return AnswerSheet(student_id=sid, exam_id="e", source_file=f"{sid}.pdf", page_count=1, answers=answers)


def test_transcription_eval_on_toy_data():
    truth = {"s1": {"Q1": "semaphore", "Q2": "a b c d"}}
    result = evaluate_transcriptions([_sheet("s1", {"Q1": "semafore", "Q2": "a b c d", "Q3": ""})], truth)
    rows = {r["question_id"]: r for r in result["rows"]}
    assert rows["Q1"]["cer"] == pytest.approx(2 / 9, abs=1e-4) and rows["Q1"]["wer"] == 1.0  # "ph" → "f": 2 edits
    assert rows["Q2"]["cer"] == 0 and rows["Q3"]["cer"] == 0  # blank vs blank is perfect
    assert result["per_student"][0]["answers"] == 3


@pytest.mark.skipif(not TRANSCRIPTS.exists(), reason="synthetic transcripts not generated yet")
def test_ground_truth_transcripts_score_zero():
    truth = load_ground_truth(TRANSCRIPTS)
    sheets = [_sheet(sid, answers) for sid, answers in truth.items()]
    result = evaluate_transcriptions(sheets, truth)
    assert result["overall"]["cer"] == 0 and result["overall"]["n"] == sum(len(a) for a in truth.values())
    assert json.dumps(result)  # JSON-serialisable, ready for output/evals/
