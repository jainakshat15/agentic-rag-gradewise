"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧱 Week 5: gradewise.grading, the grading agent                     │
│                                                                      │
│  What: one transcribed answer + its rubric + course-note evidence    │
│        → a GradeResult you can defend, criterion by criterion.       │
│      prompts.py    → the examiner prompt (handbook principles) (d1)  │
│      tools.py      → retrieve_reference · get_rubric · flag     (d2) │
│      grader.py     → grade_answer(): create_agent + guardrails  (d1) │
│      guardrails.py → clamp, injection, blanks, self-consistency (d2) │
│      graph.py      → every question in parallel with Send       (d2) │
│      events.py     → progress events for the UI                 (d4) │
│      fakes.py      → free fake models for tests and demos            │
│  ⏮ Built on: week 4 (evidence, transcription)                       │
│  ⏭ Used by: gradewise/orchestrator.py, gradewise/reports            │
╰──────────────────────────────────────────────────────────────────────╯
"""

from gradewise.grading.grader import grade_answer, make_grading_agent
from gradewise.grading.graph import build_grading_graph, grade_sheet
from gradewise.grading.guardrails import (
    DEFAULT_REVIEW_THRESHOLD,
    apply_guardrails,
    blank_result,
    detect_injection,
    enforce_rubric,
    self_consistency,
    strip_injection,
)
from gradewise.grading.tools import GRADING_TOOLS, GradingContext, flag_for_review, get_rubric, retrieve_reference

__all__ = [
    "DEFAULT_REVIEW_THRESHOLD",
    "GRADING_TOOLS",
    "GradingContext",
    "apply_guardrails",
    "blank_result",
    "build_grading_graph",
    "detect_injection",
    "enforce_rubric",
    "flag_for_review",
    "get_rubric",
    "grade_answer",
    "grade_sheet",
    "make_grading_agent",
    "retrieve_reference",
    "self_consistency",
    "strip_injection",
]
