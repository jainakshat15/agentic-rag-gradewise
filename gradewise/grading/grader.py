"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 5 · Day 1-2: Grade one answer                               │
│                                                                      │
│  What this file does                                                 │
│    Rubric + student answer + course-note evidence → GradeResult.     │
│    Two ways to get the evidence, so you can compare them:            │
│                                                                      │
│      use_tools=True  → an AGENT: the model decides what to look up   │
│                        (retrieve_reference · get_rubric · flag)      │
│      use_tools=False → a PIPELINE: we fetch evidence up front with   │
│                        find_evidence() and paste it into the prompt  │
│                                                                      │
│    Either way: create_agent(..., response_format=GradeResult), then  │
│    result["structured_response"], then the guardrails.               │
│                                                                      │
│  Use it                                                              │
│    from gradewise.grading import grade_answer                        │
│    result = grade_answer(question, answer)                           │
│                                                                      │
│  Run : uv run week5/project/grade_one.py                             │
│  ⏮ Built on: tools.py, guardrails.py, prompts.py, gradewise/rag     │
│  ⏭ Used by: graph.py (one grade_question node per question)         │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage

from gradewise.console import warn
from gradewise.errors import GradeWiseError
from gradewise.grading.guardrails import (
    DEFAULT_REVIEW_THRESHOLD,
    MAX_MODEL_CALLS,
    apply_guardrails,
    blank_result,
    detect_injection,
    grading_middleware,
    self_consistency,
    strip_injection,
    text_to_mark,
)
from gradewise.grading.prompts import EXAMINER_SYSTEM_PROMPT, build_grading_prompt
from gradewise.grading.tools import GRADING_TOOLS, GradingContext
from gradewise.schemas.answers import TranscribedAnswer
from gradewise.schemas.exam import ExamPaper, Question
from gradewise.schemas.grading import CriterionResult, GradeResult


def make_grading_agent(llm: Any = None, *, use_tools: bool = True, max_model_calls: int = MAX_MODEL_CALLS):
    """Build the grading agent ONCE; reuse it for every answer (the question travels in the context).

    💡 response_format=GradeResult: LangChain picks the strategy for you. Models with native
       structured output (OpenAI, Anthropic...) use it (ProviderStrategy); others get a hidden
       "GradeResult" tool to call (ToolStrategy). Either way the parsed, validated object lands in
       result["structured_response"].
    """
    if llm is None:
        from gradewise.models import grading_model

        llm = grading_model()
    return create_agent(
        llm,
        tools=GRADING_TOOLS if use_tools else [],
        system_prompt=EXAMINER_SYSTEM_PROMPT,
        response_format=GradeResult,
        middleware=grading_middleware(max_model_calls),
        context_schema=GradingContext,
    )


def _no_grade(question: Question, reason: str) -> GradeResult:
    """What we record when the model gave us nothing usable: 0, and a human must look."""
    return GradeResult(
        question_id=question.id,
        awarded_marks=0.0,
        max_marks=question.max_marks,
        criteria=[CriterionResult(criterion_id=c.id, awarded=0.0, justification=reason) for c in question.criteria],
        feedback="This answer could not be marked automatically; your teacher will mark it.",
        confidence=0.0,
        needs_review=True,
        review_reason=reason,
    )


def _run_once(agent, prompt: str, context: GradingContext) -> tuple[GradeResult | None, list[str], str]:
    """One agent run → (GradeResult or None, citations the tools returned, error text)."""
    try:
        state = agent.invoke({"messages": [HumanMessage(prompt)]}, context=context, config={"recursion_limit": 30})
    except GradeWiseError:
        raise  # 💡 a missing key or empty knowledge base: the learner must fix it, don't hide it
    except Exception as exc:  # noqa: BLE001
        warn(f"Grading {context.question.id} failed: {type(exc).__name__}: {exc}")
        return None, [], f"grader error ({type(exc).__name__})"
    citations = [
        c
        for m in state["messages"]
        if isinstance(m, ToolMessage) and m.name == "retrieve_reference" and isinstance(m.artifact, list)
        for c in m.artifact
    ]
    result = state.get("structured_response")
    if not isinstance(result, GradeResult):
        return None, citations, "the grader stopped without returning a grade (model-call limit?)"
    return result, citations, ""


def grade_answer(
    question: Question,
    answer: TranscribedAnswer,
    *,
    llm: Any = None,
    retriever: Any = None,
    use_tools: bool = True,
    exam: ExamPaper | None = None,
    k: int = 4,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    consistency_runs: int = 1,
    agent: Any = None,
) -> GradeResult:
    """Grade ONE transcribed answer against its question's rubric.

    Args:
        question: the question, with criteria and methodology.
        answer: the student's TranscribedAnswer (from week 4, or typed by hand).
        llm: the grading chat model (default `gradewise.models.grading_model()`).
        retriever: where evidence comes from (default: the course knowledge base, created on first use).
        use_tools: True = agent with tools; False = evidence fetched up front, then one structured call.
        exam: the whole ExamPaper (lets get_rubric answer about other questions; picks the KB collection).
        k: passages of evidence per lookup.
        review_threshold: confidence below this → needs_review.
        consistency_runs: grade this many times and take the median (1 = off). 💰 costs n× the tokens.
        agent: a pre-built agent from make_grading_agent() (the graph passes one to avoid rebuilding).

    Returns a GradeResult that has ALREADY been through the guardrails: clamped, rubric-checked, flagged.
    """
    text, notes = text_to_mark(answer)

    # 🛡️ 1. Prompt injection: detected on the raw text, removed before the model ever sees it.
    injection = detect_injection(text)
    if injection:
        text, _removed = strip_injection(text)
        notes.append("Sentences addressed to the marker were removed from this answer; mark only what remains.")

    # ⬜ 2. Blank answer (or nothing left after removing the injection): 0, decided by code.
    if not text.strip():
        if injection:
            blank = blank_result(
                question,
                "Only instructions to the marker were found: nothing to mark.",
                "Instructions to the marker are never marked. Answer the question itself to earn marks.",
            )
        else:
            blank = blank_result(question)
        return apply_guardrails(blank, question, answer, review_threshold=review_threshold, injection=injection)

    collection = exam.knowledge_base if exam else "cs201"
    context_kwargs = {"question": question, "exam": exam, "retriever": retriever, "k": k, "collection": collection}

    # 📚 3. No-tools mode: fetch the evidence ourselves and paste it in.
    evidence = None
    retrieved: list[str] | None = None
    if not use_tools:
        from gradewise.rag.evidence import find_evidence

        try:
            evidence = find_evidence(question, text, k=k, retriever=retriever, collection=collection)
        except GradeWiseError as err:
            warn(f"No evidence for {question.id}: {err.what}")
            evidence = []
            notes.append("No course-note evidence could be retrieved; mark from the rubric and model answer.")
        retrieved = [d.metadata["citation"] for d in evidence]

    prompt = build_grading_prompt(question, answer, answer_text=text, evidence=evidence, notes=notes)
    agent = agent or make_grading_agent(llm, use_tools=use_tools)

    # 🎲 4. Grade once, or n times for self-consistency.
    runs: list[GradeResult] = []
    flags: list[str] = []
    citations: list[str] = []
    error = ""
    for _ in range(max(1, consistency_runs)):
        context = GradingContext(**context_kwargs)
        result, found, error = _run_once(agent, prompt, context)
        context_kwargs["retriever"] = context.retriever  # 💡 reuse a retriever the tools created
        flags.extend(context.flags)
        citations.extend(found)
        if result is not None:
            runs.append(result)
    if not runs:
        return _no_grade(question, error or "the grader returned no grade")
    combined = self_consistency(runs, question) if len(runs) > 1 else runs[0]

    # 🧱 5. The code has the last word.
    return apply_guardrails(
        combined,
        question,
        answer,
        review_threshold=review_threshold,
        injection=injection,
        flags=list(dict.fromkeys(flags)),
        retrieved=retrieved if retrieved is not None else citations,
    )
