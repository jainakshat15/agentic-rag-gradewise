"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 5 · Day 1: What a grade looks like                          │
│                                                                      │
│  What: the grader's verdict on one answer, criterion by criterion,   │
│        with justification, evidence citations and a confidence.      │
│  ⚠️ A mark can never exceed the maximum: `clamp()` enforces it,      │
│     whatever the model says (handbook marking principle 7).          │
│  ⏭ Used by: gradewise/grading, gradewise/reports                    │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CriterionResult(BaseModel):
    """Marks awarded for one marking-scheme criterion."""

    criterion_id: str = Field(description='The criterion ID from the exam, e.g. "Q4.c2"')
    awarded: float = Field(ge=0, description="Marks awarded for this criterion")
    justification: str = Field(description="Why these marks: quote the part of the answer that earned them")
    evidence: list[str] = Field(
        default_factory=list, description="Citations from the course notes, e.g. 'unit1_operating_systems.pdf p5'"
    )


class GradeResult(BaseModel):
    """The grade for one answer."""

    question_id: str
    awarded_marks: float = Field(ge=0)
    max_marks: float = Field(gt=0)
    criteria: list[CriterionResult]
    feedback: str = Field(description="One or two sentences of constructive, student-facing feedback")
    confidence: float = Field(ge=0, le=1, description="0–1: how confident the grader is in this mark")
    needs_review: bool = Field(default=False, description="True if a human should check this mark")
    review_reason: str | None = None

    def clamp(self, criterion_max: dict[str, float] | None = None) -> GradeResult:
        """Return a copy with every mark forced inside its allowed range."""
        criteria = []
        for c in self.criteria:
            cap = (criterion_max or {}).get(c.criterion_id)
            criteria.append(c.model_copy(update={"awarded": min(c.awarded, cap)}) if cap is not None else c)
        total = sum(c.awarded for c in criteria) if criteria else self.awarded_marks
        return self.model_copy(update={"criteria": criteria, "awarded_marks": round(min(total, self.max_marks), 2)})
