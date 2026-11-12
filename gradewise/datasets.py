"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📦 gradewise/datasets.py: how much data each run uses               │
│                                                                      │
│  What this file does                                                 │
│    DATA_SIZE in your .env decides it, in ONE place:                  │
│                                                                      │
│                       full (default)        small                    │
│      knowledge base   all 4 PDFs            unit1 + handbook         │
│      exam questions   all 10 (15 parts)     Q1–Q3, all their parts   │
│      students         student_01…10         student_01, student_05   │
│      evals / golden   everything            first 5 / 2 × Q1–Q3      │
│                                                                      │
│    `small` is for people who know the ideas and just want to run     │
│    the APIs cheaply and fast. Every loader and pipeline asks this    │
│    module, so the notebooks, API, UI and CLI all agree.              │
│                                                                      │
│  Use it                                                              │
│    from gradewise import datasets                                    │
│    print(datasets.describe())          # 📦 DATA_SIZE=small: ...     │
│    datasets.kb_files()                 # the PDFs to ingest          │
│                                                                      │
│  💡 Function arguments still win: grade_batch(..., students=[...])   │
│     or ingest(files=[...]) ignore DATA_SIZE. Every helper here also  │
│     takes `small=True/False` to override the .env for one call.      │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gradewise.config import DATA_DIR, get_settings

if TYPE_CHECKING:
    from gradewise.schemas.exam import ExamPaper

KB_DIR = DATA_DIR / "knowledge-base"
SYNTHETIC_DIR = DATA_DIR / "answer-sheets" / "synthetic"
SAMPLE_EXAM = DATA_DIR / "exams" / "cs_midterm" / "expected_exam.json"

# 📦 What `small` keeps. Change these to make your own "small".
SMALL_KB_FILES: tuple[str, ...] = ("unit1_operating_systems.pdf", "cs201_course_handbook.pdf")
SMALL_QUESTIONS = 3  # the first 3 questions, with ALL their parts (Q1, Q2a, Q2b, Q3)
SMALL_STUDENTS: tuple[str, ...] = ("student_01", "student_05")  # student_05 is Sam
SMALL_EVAL_ITEMS = 5
FULL_STUDENTS: tuple[str, ...] = tuple(f"student_{i:02d}" for i in range(1, 11))


def data_size() -> str:
    """'full' or 'small' (DATA_SIZE in .env), read at call time."""
    return get_settings().data_size


def is_small(small: bool | None = None) -> bool:
    """True when DATA_SIZE=small (or when you pass small=True)."""
    return data_size() == "small" if small is None else bool(small)


# ── 📚 Knowledge base ──────────────────────────────────────────────────────
def kb_files(kb_dir: Path | str = KB_DIR, *, small: bool | None = None) -> list[Path]:
    """The knowledge-base PDFs to ingest, sorted by name.

    full → every PDF in `kb_dir`. small → only unit1_operating_systems.pdf + cs201_course_handbook.pdf.
    💡 A folder of your OWN PDFs (none of those names) is never cut down: small mode uses all of them.
    """
    pdfs = sorted(Path(kb_dir).glob("*.pdf"))
    if not is_small(small):
        return pdfs
    chosen = [p for p in pdfs if p.name in SMALL_KB_FILES]
    return chosen or pdfs


# ── 📝 Exam questions ──────────────────────────────────────────────────────
def _sample_exam() -> ExamPaper:
    from gradewise.schemas.exam import ExamPaper

    return ExamPaper.model_validate_json(SAMPLE_EXAM.read_text(encoding="utf-8"))


def question_number(question_id: str) -> str:
    """'Q2b' → '2', 'Q10a' → '10'. Sub-parts share their question's number."""
    match = re.match(r"\D*(\d+)", question_id)
    return match.group(1) if match else question_id


def exam_question_ids(exam: ExamPaper | None = None, *, small: bool | None = None) -> list[str]:
    """The IDs of the questions to grade, in paper order (default exam: the sample cs_midterm).

    full → every question. small → the first 3 questions with all their parts: ['Q1', 'Q2a', 'Q2b', 'Q3'].
    """
    paper = exam if exam is not None else _sample_exam()
    ids = [q.id for q in paper.questions]
    if not is_small(small):
        return ids
    keep: list[str] = []
    for qid in ids:
        number = question_number(qid)
        if number not in keep:
            keep.append(number)
    keep = keep[:SMALL_QUESTIONS]
    return [qid for qid in ids if question_number(qid) in keep]


def subset_exam(exam: ExamPaper, question_ids: Sequence[str] | None = None, *, small: bool | None = None) -> ExamPaper:
    """The exam with only `question_ids` (default: exam_question_ids(exam)). Totals are adjusted.

    💡 Extraction always reads the WHOLE paper (it's one call); the subset is applied after.
    """
    ids = list(question_ids) if question_ids is not None else exam_question_ids(exam, small=small)
    if ids == [q.id for q in exam.questions]:
        return exam
    unknown = set(ids) - {q.id for q in exam.questions}
    if unknown:
        raise ValueError(f"These question IDs are not on the exam {exam.exam_id!r}: {sorted(unknown)}")
    questions = [q for q in exam.questions if q.id in set(ids)]
    total = round(sum(q.max_marks for q in questions), 2)
    return exam.model_copy(update={"questions": questions, "total_marks": total})


# ── 🧑‍🎓 Synthetic students ──────────────────────────────────────────────────
def synthetic_students(*, small: bool | None = None) -> list[str]:
    """The synthetic student IDs to run. full → student_01…student_10. small → student_01 and student_05 (Sam)."""
    if is_small(small):
        return list(SMALL_STUDENTS)
    found = sorted(p.stem for p in SYNTHETIC_DIR.glob("student_*.pdf"))
    return found or list(FULL_STUDENTS)


def synthetic_sheets(*, small: bool | None = None) -> list[Path]:
    """The synthetic answer-sheet PDFs for `synthetic_students()` (only those that exist)."""
    paths = [SYNTHETIC_DIR / f"{sid}.pdf" for sid in synthetic_students(small=small)]
    return [p for p in paths if p.exists()]


# ── 🧪 Evals and golden labels ─────────────────────────────────────────────
def eval_slice[T](items: Iterable[T], *, n: int = SMALL_EVAL_ITEMS, small: bool | None = None) -> list[T]:
    """All the eval items (full), or the first `n` (small, default 5). Works on any list / iterable."""
    items = list(items)
    return items[:n] if is_small(small) else items


def golden_rows(df: Any, *, exam: ExamPaper | None = None, small: bool | None = None) -> Any:
    """Golden-label rows to compare: all (full), or the 2 small students × Q1–Q3 (small).

    `df` is a pandas DataFrame with `student_id` and `question_id` columns (e.g. data/golden/*.csv).
    """
    if not is_small(small):
        return df
    students = synthetic_students(small=True)
    questions = exam_question_ids(exam, small=True)
    mask = df["student_id"].isin(students) & df["question_id"].isin(questions)
    return df[mask].reset_index(drop=True)


# ── 📣 The banner ──────────────────────────────────────────────────────────
def _question_label(ids: Sequence[str]) -> str:
    numbers: list[str] = []
    for qid in ids:
        if question_number(qid) not in numbers:
            numbers.append(question_number(qid))
    if not numbers:
        return "no questions"
    return f"Q{numbers[0]}–Q{numbers[-1]}" if len(numbers) > 1 else f"Q{numbers[0]}"


def describe(*, small: bool | None = None) -> str:
    """One line for banners, e.g. '📦 DATA_SIZE=small: 2 PDFs · Q1–Q3 · 2 students'."""
    size = "small" if is_small(small) else "full"
    pdfs = len(kb_files(small=small))
    students = len(synthetic_students(small=small))
    try:
        ids = exam_question_ids(small=small)
        all_ids = exam_question_ids(small=False)
    except (OSError, ValueError):
        ids = all_ids = []
    if size == "small":
        questions = _question_label(ids)
    else:
        count = len({question_number(q) for q in all_ids})
        questions = f"all {count} questions ({len(all_ids)} parts)" if all_ids else "all questions"
    return f"📦 DATA_SIZE={size}: {pdfs} PDFs · {questions} · {students} students"


__all__ = [
    "SMALL_EVAL_ITEMS",
    "SMALL_KB_FILES",
    "SMALL_QUESTIONS",
    "SMALL_STUDENTS",
    "data_size",
    "describe",
    "eval_slice",
    "exam_question_ids",
    "golden_rows",
    "is_small",
    "kb_files",
    "question_number",
    "subset_exam",
    "synthetic_sheets",
    "synthetic_students",
]
