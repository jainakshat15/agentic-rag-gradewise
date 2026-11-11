"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 5 · Day 3: The report card                                  │
│                                                                      │
│  What: everything a student (and Asha) needs to see after grading.   │
│  ⏭ Used by: gradewise/reports (writer, store, Q&A agent), the UI    │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from gradewise.schemas.grading import GradeResult

# Grade bands from the CS201 course handbook (data/knowledge-base/cs201_course_handbook.pdf)
GRADE_BANDS: list[tuple[float, str]] = [(85, "A"), (70, "B"), (55, "C"), (40, "D"), (0, "F")]


def grade_band(percentage: float) -> str:
    return next(band for floor, band in GRADE_BANDS if percentage >= floor)


class ReportCard(BaseModel):
    student_id: str
    exam_id: str
    total: float
    max_total: float
    percentage: float
    grade_band: str
    results: list[GradeResult]
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    needs_review: bool = False
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_results(cls, student_id: str, exam_id: str, results: list[GradeResult], **extra) -> ReportCard:
        total = round(sum(r.awarded_marks for r in results), 2)
        max_total = round(sum(r.max_marks for r in results), 2)
        pct = round(100 * total / max_total, 1) if max_total else 0.0
        return cls(
            student_id=student_id,
            exam_id=exam_id,
            total=total,
            max_total=max_total,
            percentage=pct,
            grade_band=grade_band(pct),
            results=results,
            needs_review=any(r.needs_review for r in results),
            **extra,
        )
