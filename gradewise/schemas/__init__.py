"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 1 · Day 5: GradeWise's data contracts                       │
│                                                                      │
│  What this package does                                              │
│    Defines, with Pydantic, every object that flows through           │
│    GradeWise. Each stage hands the next one a typed, validated       │
│    object, never loose text:                                         │
│                                                                      │
│      exam.py     → ExamPaper, Question, MarkingCriterion   (wk 1, 3) │
│      answers.py  → AnswerSheet, TranscribedAnswer          (wk 4)    │
│      grading.py  → GradeResult, CriterionResult            (wk 5)    │
│      report.py   → ReportCard                              (wk 5)    │
│                                                                      │
│  💡 These models double as the `response_format` we hand to LLMs,    │
│     so the field descriptions are written FOR the model to read.     │
╰──────────────────────────────────────────────────────────────────────╯
"""

from gradewise.schemas.answers import AnswerSheet, TranscribedAnswer
from gradewise.schemas.exam import ExamPaper, MarkingCriterion, Question
from gradewise.schemas.grading import CriterionResult, GradeResult
from gradewise.schemas.report import ReportCard

__all__ = [
    "AnswerSheet",
    "CriterionResult",
    "ExamPaper",
    "GradeResult",
    "MarkingCriterion",
    "Question",
    "ReportCard",
    "TranscribedAnswer",
]
