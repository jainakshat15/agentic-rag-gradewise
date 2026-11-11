"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 5 · Day 4: Progress events                                  │
│                                                                      │
│  What this file does                                                 │
│    One tiny helper, `emit(stage, student_id, **detail)`, that any    │
│    node can call to report progress. Inside a running graph it goes  │
│    to LangGraph's "custom" stream; outside one it does nothing.      │
│                                                                      │
│      for mode, event in app.stream(inp, cfg,                         │
│                                    stream_mode=["updates", "custom"])│
│          if mode == "custom": print(event["stage"], event["detail"]) │
│                                                                      │
│  ⏭ Used by: grading/graph.py, orchestrator.py, the week 6 UI        │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from typing import Any, TypedDict


class ProgressEvent(TypedDict):
    """The shape of every custom stream event (and of `on_event(stage, student_id, detail)`)."""

    stage: str  # load_exam · transcribe · review_transcription · grade_question · grade · human_review · ...
    student_id: str | None  # None for batch-level events (e.g. load_exam)
    detail: dict[str, Any]  # always has "message" (a human-readable line); other keys depend on the stage


def emit(stage: str, student_id: str | None = None, message: str = "", **detail: Any) -> None:
    """Send a ProgressEvent to the graph's custom stream (a no-op outside a running graph).

    💡 `get_stream_writer()` only works while LangGraph is running a node. Raising outside a graph would
       make every node impossible to call directly in a notebook, so we swallow that one case.
    """
    from langgraph.config import get_stream_writer

    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(ProgressEvent(stage=stage, student_id=student_id, detail={"message": message, **detail}))
