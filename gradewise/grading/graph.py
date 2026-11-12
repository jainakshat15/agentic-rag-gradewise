"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 5 · Day 2: Grading every question in parallel               │
│                                                                      │
│  What this file does                                                 │
│    One AnswerSheet in, one ReportCard out:                           │
│                                                                      │
│        START ─▶ grade_question ×N (Send, in parallel) ─▶ aggregate   │
│                                                                      │
│    Each question is marked independently (a fresh agent run, so      │
│    Sam's brilliant Q3 can't sway the mark for his shaky Q4), and     │
│    the `results` reducer collects them all for `aggregate`.          │
│                                                                      │
│  Use it                                                              │
│    report = grade_sheet(exam, sheet)                                 │
│                                                                      │
│  ⏮ Built on: grader.py, week 3 (Send fan-out), week 4 subgraph      │
│  ⏭ Used by: gradewise/orchestrator.py                               │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from gradewise.config import max_concurrency
from gradewise.grading.events import emit
from gradewise.grading.grader import grade_answer, make_grading_agent
from gradewise.grading.guardrails import DEFAULT_REVIEW_THRESHOLD
from gradewise.schemas.answers import AnswerSheet, TranscribedAnswer
from gradewise.schemas.exam import ExamPaper, Question
from gradewise.schemas.grading import GradeResult
from gradewise.schemas.report import ReportCard

MAX_PARALLEL_QUESTIONS = 4  # 🚀 PRO: cap concurrent grading calls to stay under your provider's rate limit
#                             (MODEL_MODE=free lowers it to 2: see config.max_concurrency)


class GradingState(TypedDict, total=False):
    """What flows through the grading graph."""

    exam: ExamPaper
    sheet: AnswerSheet
    # 💡 The reducer: every parallel grade_question branch ADDS its result to this list.
    #    Without `operator.add`, the branches would overwrite each other and only one would survive.
    results: Annotated[list[GradeResult], operator.add]
    report: ReportCard


class QuestionTask(TypedDict):
    """The private input of ONE grade_question branch (sent with Send)."""

    exam: ExamPaper
    question: Question
    answer: TranscribedAnswer
    student_id: str


def missing_answer(question_id: str) -> TranscribedAnswer:
    """A question with no answer on the sheet is a blank answer: certain, and worth 0."""
    return TranscribedAnswer(question_id=question_id, text="", confidence=1.0)


def build_grading_graph(
    llm: Any = None,
    retriever: Any = None,
    *,
    consistency_runs: int = 1,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    use_tools: bool = True,
    k: int = 4,
):
    """Build and compile the grading graph. `llm` defaults to `gradewise.models.grading_model()`.

    Input:  {"exam": ExamPaper, "sheet": AnswerSheet}
    Output: the same state plus "results" (one GradeResult per question) and "report" (a ReportCard).

    💡 The agent is built ONCE here (a missing API key fails fast), and shared by every branch:
       the question to mark travels in each run's context, not in the agent.
    """
    agent = make_grading_agent(llm, use_tools=use_tools)

    def fan_out(state: GradingState) -> list[Send]:
        sheet = state["sheet"]
        return [
            Send(
                "grade_question",
                QuestionTask(
                    exam=state["exam"],
                    question=q,
                    answer=sheet.answer_for(q.id) or missing_answer(q.id),
                    student_id=sheet.student_id,
                ),
            )
            for q in state["exam"].questions
        ]

    def grade_question(task: QuestionTask) -> dict:
        result = grade_answer(
            task["question"],
            task["answer"],
            agent=agent,
            retriever=retriever,
            use_tools=use_tools,
            exam=task["exam"],
            k=k,
            review_threshold=review_threshold,
            consistency_runs=consistency_runs,
        )
        emit(
            "grade_question",
            task["student_id"],
            f"{result.question_id}: {result.awarded_marks:g}/{result.max_marks:g}"
            + (" ⚑" if result.needs_review else ""),
            question_id=result.question_id,
            awarded=result.awarded_marks,
            max=result.max_marks,
            needs_review=result.needs_review,
        )
        return {"results": [result]}

    def aggregate(state: GradingState) -> dict:
        # 💡 Parallel branches finish in any order. Put the results back in exam order, and if a
        #    question was somehow graded twice keep the latest, so the total is always right.
        by_id = {r.question_id: r for r in state["results"]}
        ordered = [by_id[q.id] for q in state["exam"].questions if q.id in by_id]
        sheet = state["sheet"]
        report = ReportCard.from_results(sheet.student_id, state["exam"].exam_id, ordered)
        emit("grade", sheet.student_id, f"{report.total:g}/{report.max_total:g} ({report.grade_band})")
        return {"report": report}

    graph = StateGraph(GradingState)
    graph.add_node("grade_question", grade_question)
    graph.add_node("aggregate", aggregate)
    graph.add_conditional_edges(START, fan_out, ["grade_question"])
    graph.add_edge("grade_question", "aggregate")  # 💡 waits for EVERY branch before aggregating
    graph.add_edge("aggregate", END)
    # 💡 checkpointer=False: when the orchestrator runs this graph inside one of its nodes, don't
    #    checkpoint each question separately. The orchestrator saves the finished ReportCard instead.
    return graph.compile(checkpointer=False)


def grade_sheet(
    exam: ExamPaper,
    sheet: AnswerSheet,
    *,
    llm: Any = None,
    retriever: Any = None,
    consistency_runs: int = 1,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    use_tools: bool = True,
    graph: Any = None,
    questions: Sequence[str] | None = None,
) -> ReportCard:
    """Grade a whole AnswerSheet and return its ReportCard (strengths/improvements are filled in later).

    📦 questions: only grade these question IDs. Default: all, or Q1–Q3 when DATA_SIZE=small.
    """
    from gradewise import datasets

    if questions is not None or datasets.is_small():
        exam = datasets.subset_exam(exam, questions)
    app = graph or build_grading_graph(
        llm, retriever, consistency_runs=consistency_runs, review_threshold=review_threshold, use_tools=use_tools
    )
    config = {"max_concurrency": max_concurrency(MAX_PARALLEL_QUESTIONS)}
    state = app.invoke({"exam": exam, "sheet": sheet}, config=config)
    return state["report"]
