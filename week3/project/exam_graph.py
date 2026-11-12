"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 3 · Day 5: 🧱 PROJECT: exam understanding v2, as a graph    │
│                                                                      │
│  What this script does                                               │
│    The week 1 extractor, rebuilt as a LangGraph workflow that        │
│    checks its own work and asks Emma before saving anything:         │
│                                                                      │
│      parse → extract → validate ─┬─ problems? → retry → extract      │
│                                  └─ clean / out of tries             │
│                                      → admin_review ⏸ → store        │
│                                                                      │
│  Run:  uv run week3/project/exam_graph.py                 (asks you) │
│        uv run week3/project/exam_graph.py --auto-approve             │
│        uv run week3/project/exam_graph.py --sqlite   (pause & resume │
│                                                       across runs)   │
│                                                                      │
│  ⏮ Built on: week1/project/extract_exam.py, week3/day4 (interrupt)   │
│  ⏭ Used by: gradewise/exam/graph.py (the packaged version)           │
╰──────────────────────────────────────────────────────────────────────╯
"""

import argparse
import json
import sqlite3
import sys
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from gradewise.config import DATA_DIR, OUTPUT_DIR
from gradewise.console import banner, console, step, success, tip, warn
from gradewise.errors import GradeWiseError
from gradewise.exam.extract import build_messages  # 🔁 the same prompt as the packaged extractor
from gradewise.exam.parse import parse_pdf
from gradewise.exam.summary import print_exam_summary
from gradewise.models import chat_model
from gradewise.schemas import ExamPaper

PDFS = ["question_paper", "marking_scheme", "methodology"]
LLM = {"model": None}  # 💡 created once in main(), so a missing key fails before any work is done


# ── 1. The state: what the graph remembers between nodes ─────────────────
class ExamState(TypedDict, total=False):
    exam_id: str
    documents: dict[str, str]  # name → clean text
    paper: dict[str, Any] | None  # 💡 a dict, so any checkpointer can save it
    problems: list[str]
    attempts: int
    max_attempts: int
    approved: bool


# ── 2. The nodes: plain functions, state in → partial update out ─────────
def parse(state: ExamState) -> dict:
    folder = DATA_DIR / "exams" / state["exam_id"]
    docs = {name: parse_pdf(folder / f"{name}.pdf").text for name in PDFS}
    if (folder / "model_answers.md").exists():
        docs["model_answers"] = (folder / "model_answers.md").read_text(encoding="utf-8")
    step(f"Parsed {len(docs)} documents from {folder.name}/")
    # ⚠️ With --sqlite the thread remembers the last run, so we reset the per-run fields here.
    return {"documents": docs, "attempts": 0, "problems": [], "paper": None, "approved": False}


def extract(state: ExamState) -> dict:
    attempt = state.get("attempts", 0) + 1
    step(f"Extracting (attempt {attempt} of {state['max_attempts']})…")
    docs = state["documents"]
    messages = build_messages(
        docs["question_paper"],
        docs["marking_scheme"],
        docs["methodology"],
        exam_id=state["exam_id"],
        model_answers=docs.get("model_answers"),
        previous=state.get("paper"),
        feedback=state.get("problems") or None,  # 💡 on a retry, the model sees its own mistakes
    )
    try:
        paper = LLM["model"].with_structured_output(ExamPaper).invoke(messages)
    except ValueError as exc:  # 💡 includes Pydantic's ValidationError: the JSON didn't fit the schema
        return {"attempts": attempt, "problems": [f"Output didn't match the schema: {exc}"]}
    paper = paper.model_copy(update={"exam_id": state["exam_id"]})
    return {"attempts": attempt, "paper": paper.model_dump()}


def validate(state: ExamState) -> dict:
    if not state.get("paper"):
        return {"problems": state.get("problems") or ["No exam extracted."]}
    # 💡 Pydantic checked the shape; problems() checks the arithmetic (marks add up?)
    return {"problems": ExamPaper.model_validate(state["paper"]).problems()}


def retry(state: ExamState) -> dict:
    for problem in state["problems"]:
        warn(problem)
    tip("Sending these back to the model so it can fix them.")
    return {}


def admin_review(state: ExamState) -> dict:
    # ⚠️ When we resume, this node runs again FROM THE TOP and interrupt() returns the answer.
    #    So never do anything with side effects (like saving) before the interrupt() line.
    decision = interrupt({"paper": state.get("paper"), "problems": state.get("problems", [])})
    # ✋ TODO (week3_EXERCISE): let Emma send back {"edits": {...}} and apply them here.
    return {"approved": bool(decision.get("approved")) and bool(state.get("paper"))}


def store(state: ExamState) -> dict:
    from gradewise.db import save_exam  # 💡 the SQLite database in output/gradewise.db

    paper = ExamPaper.model_validate(state["paper"])
    save_exam(paper, approved_by="Emma")
    out = OUTPUT_DIR / f"{paper.exam_id}.json"
    out.write_text(json.dumps(paper.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    success("Exam approved and stored", {"Database": "output/gradewise.db", "JSON": out})
    return {}


# ── 3. The routing: where to go after validate and after review ──────────
def after_validate(state: ExamState) -> str:
    if state["problems"] and state["attempts"] < state["max_attempts"]:
        return "retry"
    return "admin_review"  # 💡 clean, OR out of tries: either way a human decides


def after_review(state: ExamState) -> str:
    return "store" if state.get("approved") else END


# ── 4. Wire it up ────────────────────────────────────────────────────────
def build_graph(checkpointer):
    builder = StateGraph(ExamState)
    for node in (parse, extract, validate, retry, admin_review, store):
        builder.add_node(node.__name__, node)
    builder.add_edge(START, "parse")
    builder.add_edge("parse", "extract")
    builder.add_edge("extract", "validate")
    builder.add_conditional_edges("validate", after_validate, ["retry", "admin_review"])
    builder.add_edge("retry", "extract")
    builder.add_conditional_edges("admin_review", after_review, ["store", END])
    builder.add_edge("store", END)
    return builder.compile(checkpointer=checkpointer)  # ⚠️ interrupt() needs a checkpointer


def ask_emma(payload: dict, auto_approve: bool) -> dict:
    """Show the extracted exam and get a decision (from the keyboard, or automatically)."""
    if payload["paper"]:
        print_exam_summary(ExamPaper.model_validate(payload["paper"]), payload["problems"])
    else:
        warn("No valid exam could be extracted.")
    if auto_approve:
        tip("--auto-approve is on: approving without asking.")
        return {"approved": True}
    answer = input("🧑‍🏫 Emma, approve this exam? [y/N] ").strip().lower()
    return {"approved": answer in {"y", "yes"}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Exam understanding v2: extract → validate → retry → approve")
    parser.add_argument("--exam-id", default="cs_midterm")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--auto-approve", action="store_true", help="approve without asking (for demos and CI)")
    parser.add_argument("--sqlite", action="store_true", help="save progress in output/exam_checkpoints.db")
    args = parser.parse_args()
    banner("Week 3 · Day 5", "Exam understanding v2: a graph with a human in the loop")
    LLM["model"] = chat_model(temperature=0)  # ⚠️ raises a friendly MissingAPIKeyError if your key isn't set

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.sqlite:
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(str(OUTPUT_DIR / "exam_checkpoints.db"), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    else:
        checkpointer = InMemorySaver()
    graph = build_graph(checkpointer)
    config = {"configurable": {"thread_id": f"exam-{args.exam_id}"}}  # 💡 one thread per exam

    # 💡 With --sqlite, a review you left paused last time is still waiting: pick it up.
    if graph.get_state(config).next == ("admin_review",):
        tip("Found a paused review from last time. Resuming it.")
        result = {"__interrupt__": graph.get_state(config).tasks[0].interrupts}
    else:
        result = graph.invoke({"exam_id": args.exam_id, "max_attempts": args.max_attempts}, config)

    while "__interrupt__" in result:  # ⏸ the graph is paused, waiting for Emma
        decision = ask_emma(result["__interrupt__"][0].value, args.auto_approve)
        result = graph.invoke(Command(resume=decision), config)

    if not result.get("approved"):
        warn("Not approved, so nothing was stored. Run again when you're ready.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GradeWiseError as err:
        err.show()
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n👋 Stopped. With --sqlite, your progress is saved for next time.")
        sys.exit(130)
