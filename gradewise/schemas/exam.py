"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 1 · Day 5: The exam schema                                  │
│                                                                      │
│  What: the JSON shape of an exam. A question paper + marking scheme  │
│        + methodology become ONE ExamPaper: an array of Questions,    │
│        each with its criteria and how to mark it.                    │
│  ⏮ Built on: week1/day3.ipynb (structured output)                    │
│  ⏭ Used by: gradewise/exam (extraction), gradewise/grading           │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

QuestionType = Literal["definition", "short", "long", "compare", "numerical", "mcq"]


class MarkingCriterion(BaseModel):
    """One mark-earning point from the marking scheme."""

    id: str = Field(description='Stable ID: question ID + ".c" + number, e.g. "Q4.c2"')
    description: str = Field(description="What the student must say or do to earn these marks")
    marks: float = Field(gt=0, description="Marks available for this criterion")
    keywords: list[str] = Field(default_factory=list, description="Helpful terms, NOT hard requirements")
    partial_credit_rule: str | None = Field(default=None, description="When fewer than full marks may be given")


class Question(BaseModel):
    """One question (or sub-part) on the paper, with everything needed to mark it."""

    id: str = Field(description='Stable ID, e.g. "Q4" or "Q4b" for a sub-part')
    number: str = Field(description='Number as printed on the paper, e.g. "4(b)"')
    text: str = Field(description="The full question text, exactly as printed")
    max_marks: float = Field(gt=0)
    question_type: QuestionType
    criteria: list[MarkingCriterion] = Field(min_length=1)
    methodology: str = Field(description="The examiner's instructions for checking this answer")
    reference_topics: list[str] = Field(
        default_factory=list, description="Course topics to retrieve evidence for, e.g. ['deadlock conditions']"
    )
    model_answer: str | None = None

    @property
    def criteria_total(self) -> float:
        return round(sum(c.marks for c in self.criteria), 2)


class ExamPaper(BaseModel):
    """A whole exam: the array of questions GradeWise marks against."""

    exam_id: str = Field(description='Short slug, e.g. "cs_midterm"')
    title: str
    subject: str
    knowledge_base: str = Field(default="cs201", description="Which knowledge-base collection to retrieve from")
    total_marks: float = Field(gt=0)
    questions: list[Question] = Field(min_length=1)

    @property
    def questions_total(self) -> float:
        return round(sum(q.max_marks for q in self.questions), 2)

    @model_validator(mode="after")
    def _unique_ids(self) -> ExamPaper:
        ids = [q.id for q in self.questions]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"Duplicate question IDs: {sorted(dupes)}")
        return self

    def problems(self) -> list[str]:
        """Consistency checks that are *not* fatal at parse time. The week 3 validator loops on these.

        💡 We keep them separate from Pydantic validation on purpose: an LLM extraction
           that fails here can be sent back to the model WITH this list, so it can fix itself.
        """
        issues: list[str] = []
        if abs(self.questions_total - self.total_marks) > 0.01:
            issues.append(
                f"Question marks add up to {self.questions_total}, but the paper total is {self.total_marks}."
            )
        for q in self.questions:
            if abs(q.criteria_total - q.max_marks) > 0.01:
                issues.append(
                    f"{q.id}: criteria add up to {q.criteria_total}, but the question is worth {q.max_marks}."
                )
            if not q.methodology.strip():
                issues.append(f"{q.id}: no marking methodology.")
        return issues
