"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧱 Week 4 · Day 5: The transcription subgraph                       │
│                                                                      │
│  What: one answer-sheet PDF in, one AnswerSheet out:                 │
│                                                                      │
│     render ─▶ transcribe_page ×N (Send, in parallel) ─▶ map          │
│            ─▶ verify_low_confidence ─▶ END                           │
│                                                                      │
│  Run:  transcribe_sheet("student_03.pdf", exam, "student_03")        │
│  ⏮ Built on: week3 (StateGraph, Send fan-out)                       │
│  ⏭ Used by: week4/project/transcription_graph.py, week5 grading     │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import operator
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from PIL import Image

from gradewise.console import step, success, tip, warn
from gradewise.schemas.answers import AnswerSheet
from gradewise.schemas.exam import ExamPaper
from gradewise.transcription.confidence import (
    MAX_PASS_CER,
    REVIEW_THRESHOLD,
    VERIFY_DPI,
    VerificationCheck,
    double_pass_verify,
    flagged_answers,
    needs_review,
)
from gradewise.transcription.mapper import map_answers
from gradewise.transcription.pdf_to_images import DEFAULT_DPI, estimate_image_tokens, preprocess, render_pages
from gradewise.transcription.schemas import PageTranscription
from gradewise.transcription.vision import transcribe_page_with_usage

# 💰 ROUGH prices in USD per million tokens, ONLY for the cost estimate we print.
# ⚠️ Illustrative numbers for a mid-tier vision model: check your provider's pricing page and
#    edit them. Nothing is billed from these; they just help you compare DPI settings.
ROUGH_PRICE_PER_M_INPUT = 1.25
ROUGH_PRICE_PER_M_OUTPUT = 10.0
MAX_PARALLEL_PAGES = 4  # 🚀 PRO: cap concurrent vision calls so a 12-page script doesn't hit rate limits


class TranscriptionState(TypedDict, total=False):
    """What flows through the subgraph."""

    pdf_path: str
    exam: ExamPaper
    student_id: str
    dpi: int
    images: list[Image.Image]
    # 💡 Annotated reducers: each parallel `transcribe_page` branch ADDS its page, instead of
    #    overwriting the others' work. This is what makes the Send fan-out safe.
    pages: Annotated[list[PageTranscription], operator.add]
    usage: Annotated[list[dict[str, int]], operator.add]
    sheet: AnswerSheet
    checks: list[VerificationCheck]


class PageTask(TypedDict):
    """The private input of ONE transcribe_page branch (sent with Send)."""

    image: Image.Image
    page_number: int
    question_ids: list[str]


def build_transcription_graph(
    llm: Any = None,
    *,
    verify: bool = True,
    verify_dpi: int = VERIFY_DPI,
    threshold: float = REVIEW_THRESHOLD,
    max_cer: float = MAX_PASS_CER,
):
    """Build and compile the transcription subgraph. `llm` defaults to `gradewise.models.vision_model()`.

    💡 The model is created ONCE here, so a missing API key fails fast, before any page is rendered.
    """
    if llm is None:
        from gradewise.models import vision_model

        llm = vision_model()

    def render(state: TranscriptionState) -> dict:
        dpi = state.get("dpi", DEFAULT_DPI)
        step(f"Rendering {Path(state['pdf_path']).name} at {dpi} DPI…")
        images = [preprocess(img) for img in render_pages(state["pdf_path"], dpi=dpi)]
        return {"images": images}

    def fan_out(state: TranscriptionState) -> list[Send]:
        # 💡 One Send per page → LangGraph runs these branches in parallel (same super-step).
        question_ids = [q.id for q in state["exam"].questions]
        return [
            Send("transcribe_page", {"image": img, "page_number": i, "question_ids": question_ids})
            for i, img in enumerate(state["images"], start=1)
        ]

    def transcribe_page_node(task: PageTask) -> dict:
        page, usage = transcribe_page_with_usage(task["image"], task["page_number"], task["question_ids"], llm)
        if not usage["input_tokens"]:
            # 💡 Some providers/fakes don't report usage: fall back to our pixel-based estimate.
            usage = {"input_tokens": estimate_image_tokens(task["image"]) + 900, "output_tokens": 0, "estimated": 1}
        return {"pages": [page], "usage": [usage]}

    def map_node(state: TranscriptionState) -> dict:
        sheet = map_answers(
            state["pages"],
            state["exam"],
            state["student_id"],
            source_file=Path(state["pdf_path"]).name,
            page_count=len(state["images"]),
        )
        answered = sum(1 for a in sheet.answers if not a.is_blank)
        step(f"Mapped {answered}/{len(sheet.answers)} answers · {len(sheet.unmapped_text)} unmapped snippets")
        return {"sheet": sheet}

    def verify_node(state: TranscriptionState) -> dict:
        sheet = state["sheet"]
        if not verify:
            return {"checks": []}
        low = [a.question_id for a in sheet.answers if a.pages and needs_review(a, threshold)]
        if low:
            step(f"Double-checking {len(low)} low-confidence answer(s): {', '.join(low)}")
        sheet, checks = double_pass_verify(
            sheet,
            state["pages"],
            state["exam"],
            llm=llm,
            pdf_path=state["pdf_path"],
            verify_dpi=verify_dpi,
            threshold=threshold,
            max_cer=max_cer,
        )
        return {"sheet": sheet, "checks": checks}

    graph = StateGraph(TranscriptionState)
    graph.add_node("render", render)
    graph.add_node("transcribe_page", transcribe_page_node)
    graph.add_node("map", map_node)
    graph.add_node("verify_low_confidence", verify_node)
    graph.add_edge(START, "render")
    graph.add_conditional_edges("render", fan_out, ["transcribe_page"])
    graph.add_edge("transcribe_page", "map")  # 💡 waits for ALL branches before mapping
    graph.add_edge("map", "verify_low_confidence")
    graph.add_edge("verify_low_confidence", END)
    return graph.compile()


def log_cost(usage: list[dict[str, int]], pages: int, seconds: float) -> dict[str, Any]:
    """Print a rough per-sheet token and cost summary, and return it."""
    tokens_in = sum(u.get("input_tokens", 0) for u in usage)
    tokens_out = sum(u.get("output_tokens", 0) for u in usage)
    cost = tokens_in / 1e6 * ROUGH_PRICE_PER_M_INPUT + tokens_out / 1e6 * ROUGH_PRICE_PER_M_OUTPUT
    estimated = any(u.get("estimated") for u in usage)
    summary = {
        "Pages": pages,
        "Vision calls (first pass)": len(usage),
        "Input tokens": f"{tokens_in:,}" + (" (estimated)" if estimated else ""),
        "Output tokens": f"{tokens_out:,}",
        "Rough cost": f"${cost:.4f}",
        "Wall time": f"{seconds:.1f}s",
    }
    return summary


def transcribe_sheet(
    pdf_path: str | Path,
    exam: ExamPaper,
    student_id: str,
    llm: Any = None,
    *,
    dpi: int = DEFAULT_DPI,
    verify: bool = True,
    threshold: float = REVIEW_THRESHOLD,
    quiet: bool = False,
) -> AnswerSheet:
    """Transcribe one handwritten answer-sheet PDF into an AnswerSheet (one answer per exam question)."""
    app = build_transcription_graph(llm, verify=verify, threshold=threshold)
    start = time.perf_counter()
    state = app.invoke(
        {"pdf_path": str(pdf_path), "exam": exam, "student_id": student_id, "dpi": dpi},
        config={"max_concurrency": MAX_PARALLEL_PAGES},
    )
    sheet: AnswerSheet = state["sheet"]
    if not quiet:
        summary = log_cost(state.get("usage", []), sheet.page_count, time.perf_counter() - start)
        flagged = flagged_answers(sheet, threshold)
        summary["Needs review"] = ", ".join(flagged) or "none"
        success(f"Transcribed {student_id}", summary, next_step="Week 5 grades these answers")
        if flagged:
            warn(f"{len(flagged)} answer(s) are below {threshold:.2f} confidence: a human should check them.")
        if dpi > DEFAULT_DPI:
            tip("Higher DPI rarely helps after the size cap; compare CER at 150 vs 200 DPI before paying for 300.")
    return sheet
