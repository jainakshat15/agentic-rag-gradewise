"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 3 · Day 5: Exam understanding v2, as a graph                │
│                                                                      │
│  What this file does                                                 │
│    The week 1 chain, rebuilt as a LangGraph workflow that checks     │
│    its own work and asks Emma before saving anything:                │
│                                                                      │
│      parse → extract → validate ─┬─ problems? → retry → extract      │
│                                  └─ clean / out of tries             │
│                                        → admin_review ⏸ → store      │
│                                                                      │
│  Use it                                                              │
│    graph = build_exam_graph(checkpointer="sqlite")                   │
│    cfg = {"configurable": {"thread_id": "cs_midterm"}}               │
│    graph.invoke(start_input("cs_midterm"), cfg)    # pauses ⏸         │
│    graph.invoke(Command(resume={"approved": True}), cfg)             │
│                                                                      │
│  ⏮ Built on: week3/day4.ipynb (checkpointers, interrupt)             │
│  ⏭ Used by: gradewise CLI `exam extract`, week 6 API & dashboard     │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from gradewise.config import OUTPUT_DIR
from gradewise.console import step, success, warn
from gradewise.errors import DataNotFoundError
from gradewise.exam.extract import PACK_FILES, exam_pack, extract_once
from gradewise.exam.parse import read_document
from gradewise.exam.summary import exam_summary, print_exam_summary
from gradewise.schemas.exam import ExamPaper

CHECKPOINT_DB = OUTPUT_DIR / "exam_checkpoints.db"

__all__ = [
    "Command",
    "ExamState",
    "build_exam_graph",
    "exam_summary",
    "print_exam_summary",
    "start_input",
    "sqlite_checkpointer",
]


class ExamState(TypedDict, total=False):
    """Everything the workflow knows, saved by the checkpointer after every node.

    💡 We keep the exam as a plain dict (not an ExamPaper object) so every checkpointer can
       serialise it; nodes turn it back into an ExamPaper when they need to.
    """

    exam_id: str
    sources: dict[str, str]  # question_paper / marking_scheme / methodology (/ model_answers) → path
    documents: dict[str, str]  # the same keys → clean text (pass these to skip parsing)
    paper: dict[str, Any] | None  # the latest ExamPaper, as a dict
    problems: list[str]  # what's wrong with it (empty = clean)
    attempts: int
    max_attempts: int
    approved: bool | None
    stored: bool


def start_input(
    exam_id: str = "cs_midterm",
    *,
    max_attempts: int = 3,
    sources: dict[str, str] | None = None,
    documents: dict[str, str] | None = None,
) -> ExamState:
    """The input for a NEW run: `graph.invoke(start_input("cs_midterm"), config)`.

    💡 Every key is set explicitly (even to None), so a re-run on a thread that the checkpointer
       remembers re-reads the PDFs instead of reusing last time's text.
    sources:   paths to parse (default: data/exams/<exam_id>/). documents: text, to skip parsing.
    """
    return {"exam_id": exam_id, "max_attempts": max_attempts, "sources": sources, "documents": documents}


def sqlite_checkpointer(path: str | Path | None = None):
    """A SqliteSaver in output/ so a paused review survives restarting Python."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(path) if path else CHECKPOINT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    # ⚠️ check_same_thread=False: LangGraph may run nodes on another thread.
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def build_exam_graph(
    checkpointer: BaseCheckpointSaver | Literal["memory", "sqlite"] | None = None,
    llm=None,
    *,
    db: Any = None,
    write_json: bool = True,
):
    """Compile the exam-understanding workflow.

    checkpointer: a saver, "sqlite" (output/exam_checkpoints.db) or "memory"/None (in-memory).
                  ⚠️ interrupt() needs *some* checkpointer, so there is always one.
    llm:          the chat model; defaults to gradewise.models.chat_model() on first use.
    db:           where `store` saves the exam (see gradewise.db); None = output/gradewise.db.
    write_json:   also write output/<exam_id>.json when storing.
    """
    model = {"llm": llm}

    def get_llm():
        if model["llm"] is None:
            from gradewise.models import chat_model

            model["llm"] = chat_model(temperature=0)
        return model["llm"]

    # ── nodes ────────────────────────────────────────────────────────────
    def parse(state: ExamState) -> dict:
        # ⚠️ A checkpointed thread remembers the LAST run's values. Running the same thread again
        #    would otherwise start with attempts=3 and old problems, so every run starts clean here.
        fresh = {"attempts": 0, "problems": [], "paper": None, "approved": None, "stored": False}
        if state.get("documents"):
            return fresh  # 💡 text was handed in directly: nothing to parse
        exam_id = state.get("exam_id", "cs_midterm")
        sources = state.get("sources") or {k: str(p) for k, p in exam_pack(exam_id).items()}
        step(f"Parsing {len(sources)} documents…")
        missing = [key for key in PACK_FILES if key not in sources]
        if missing:
            raise DataNotFoundError(f"The exam sources are missing: {', '.join(missing)}.")
        # 💡 model_answers (optional) comes along too when the pack has it
        return fresh | {"documents": {key: read_document(Path(path)) for key, path in sources.items()}}

    def extract(state: ExamState) -> dict:
        attempt = state.get("attempts", 0) + 1
        step(f"Extracting the exam (attempt {attempt}/{state.get('max_attempts', 3)})…")
        docs = state["documents"]
        paper, problems = extract_once(
            docs["question_paper"],
            docs["marking_scheme"],
            docs["methodology"],
            exam_id=state.get("exam_id", "cs_midterm"),
            llm=get_llm(),
            model_answers=docs.get("model_answers"),
            previous=state.get("paper"),
            feedback=state.get("problems") or None,  # 🔁 last attempt's problems → self-correction
        )
        update: dict[str, Any] = {"attempts": attempt, "problems": problems}
        if paper is not None:
            update["paper"] = paper.model_dump()
        return update

    def validate(state: ExamState) -> dict:
        # 💡 extract_once already ran ExamPaper.problems(); this node makes the check visible
        #    in the graph (and in LangSmith traces) and is the one place to add more rules.
        if not state.get("paper"):
            return {"problems": state.get("problems") or ["No valid ExamPaper was produced."]}
        problems = ExamPaper.model_validate(state["paper"]).problems()
        if not problems:
            step("Validation passed: every mark is accounted for.")
        return {"problems": problems}

    def retry(state: ExamState) -> dict:
        warn(f"{len(state['problems'])} problem(s) found. Sending them back to the model…")
        return {}

    def admin_review(state: ExamState) -> dict:
        # ⚠️ On resume, LangGraph re-runs this node FROM THE TOP, and interrupt() then returns
        #    Emma's answer. So nothing before interrupt() may have side effects (no saving!).
        paper = state.get("paper")
        decision = interrupt(
            {
                "message": "Please review the extracted exam. Resume with {'approved': bool, 'edits': {...}}.",
                "summary": exam_summary(ExamPaper.model_validate(paper)) if paper else None,
                "problems": state.get("problems", []),
                "paper": paper,
            }
        )
        decision = decision if isinstance(decision, dict) else {"approved": bool(decision)}
        update: dict[str, Any] = {"approved": bool(decision.get("approved"))}
        if decision.get("edits"):
            # 💡 Emma may hand back a patched exam: it must still pass the schema.
            try:
                edited = ExamPaper.model_validate({**(paper or {}), **decision["edits"]})
            except ValidationError as exc:
                warn(f"The edited exam isn't valid, so it was not approved: {exc.error_count()} error(s).")
                return {"approved": False, "problems": [str(e["msg"]) for e in exc.errors()]}
            update |= {"paper": edited.model_dump(), "problems": edited.problems()}
        elif paper is None and update["approved"]:
            warn("There is no valid exam to approve. Provide one in 'edits'.")
            update["approved"] = False
        return update

    def store(state: ExamState) -> dict:
        from gradewise.db import save_exam  # 💡 imported here so building the graph needs no database

        paper = ExamPaper.model_validate(state["paper"])
        save_exam(paper, approved_by="admin", db=db)
        details: dict[str, Any] = {"Exam": paper.exam_id, "Questions": len(paper.questions), "Marks": paper.total_marks}
        if write_json:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out = OUTPUT_DIR / f"{paper.exam_id}.json"
            out.write_text(json.dumps(paper.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
            details["JSON"] = out
        success("Exam approved and stored", details, next_step="Week 4: read the handwritten answer sheets")
        return {"stored": True}

    # ── edges ────────────────────────────────────────────────────────────
    def after_validate(state: ExamState) -> str:
        if not state.get("problems"):
            return "admin_review"
        if state.get("attempts", 0) < state.get("max_attempts", 3):
            return "retry"
        return "admin_review"  # 💡 out of tries: a human decides, problems and all

    def after_review(state: ExamState) -> str:
        return "store" if state.get("approved") else END

    builder = StateGraph(ExamState)
    for name, node in [
        ("parse", parse),
        ("extract", extract),
        ("validate", validate),
        ("retry", retry),
        ("admin_review", admin_review),
        ("store", store),
    ]:
        builder.add_node(name, node)
    builder.add_edge(START, "parse")
    builder.add_edge("parse", "extract")
    builder.add_edge("extract", "validate")
    builder.add_conditional_edges("validate", after_validate, ["retry", "admin_review"])
    builder.add_edge("retry", "extract")
    builder.add_conditional_edges("admin_review", after_review, ["store", END])
    builder.add_edge("store", END)

    if checkpointer is None or checkpointer == "memory":
        checkpointer = InMemorySaver()
    elif checkpointer == "sqlite":
        checkpointer = sqlite_checkpointer()
    return builder.compile(checkpointer=checkpointer)
