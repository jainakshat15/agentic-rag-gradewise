"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧱 Week 5 · Day 4: The orchestrator, GradeWise end to end           │
│                                                                      │
│  What this file does                                                 │
│    One graph that takes a pile of answer sheets to stored report     │
│    cards, pausing for Asha whenever a human should decide:           │
│                                                                      │
│    load_exam ─▶ transcribe ×students (Send) ─▶ review_transcription  │
│      ─▶ grade ×students (Send; each runs the grading graph)          │
│      ─▶ aggregate ─▶ human_review (only if something is flagged)     │
│      ─▶ write_report ─▶ store ─▶ END                                 │
│                                                                      │
│    ⏸ review_transcription and human_review call interrupt(). With a │
│      SqliteSaver checkpointer the paused run lives in a file, so     │
│      Asha can close her laptop and resume tomorrow.                  │
│                                                                      │
│  Use it                                                              │
│    result = grade_batch("cs_midterm", pdfs_or_sheets, thread_id="b1")│
│                                                                      │
│  Run : uv run week5/project/orchestrator.py --students 3             │
│  ⏮ Built on: transcription/ (wk 4), grading/ + reports/ (wk 5)      │
│  ⏭ Used by: gradewise/cli.py `grade`, the week 6 API and UI         │
╰──────────────────────────────────────────────────────────────────────╯

📡 PROGRESS EVENTS (for the week 6 UI)
   Every node reports progress with `emit(stage, student_id, message, **extra)` (gradewise/grading/events.py),
   which writes to LangGraph's "custom" stream. Two ways to consume it:

   1. grade_batch(..., on_event=callback): callback(stage, student_id, detail) is called for every event,
      in order, while the batch runs. `detail` is a dict that always has "message".
   2. Stream the graph yourself:
          for namespace, mode, chunk in app.stream(inp, cfg, stream_mode=["updates", "custom"], subgraphs=True):
              if mode == "custom":   ...   # chunk = {"stage", "student_id", "detail"}
              if mode == "updates" and "__interrupt__" in chunk:  ...   # paused for a human

   Stages, in order: load_exam · transcribe · review_transcription · grade · grade_question (one per question)
   · aggregate · human_review · write_report · store · done, plus paused / resumed from grade_batch itself.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from gradewise.config import DATA_DIR, OUTPUT_DIR
from gradewise.console import tip
from gradewise.db import DB, ExamNotFoundError
from gradewise.db import load_exam as db_load_exam
from gradewise.errors import DataNotFoundError
from gradewise.grading.events import emit
from gradewise.grading.graph import MAX_PARALLEL_QUESTIONS, build_grading_graph
from gradewise.grading.guardrails import DEFAULT_REVIEW_THRESHOLD
from gradewise.reports.store import store_report
from gradewise.reports.writer import rule_based_feedback, save_report_files, write_feedback
from gradewise.schemas.answers import AnswerSheet, TranscribedAnswer
from gradewise.schemas.exam import ExamPaper
from gradewise.schemas.report import ReportCard

CHECKPOINT_DB = OUTPUT_DIR / "grading_checkpoints.db"
MAX_PARALLEL_STUDENTS = 3  # 🚀 PRO: students × questions run at once can hit rate limits; tune with care

EventCallback = Callable[[str, str | None, dict[str, Any]], None]
Reviewer = Callable[[dict[str, Any]], dict[str, Any]]

__all__ = [
    "BatchResult",
    "Command",
    "apply_review",
    "apply_transcription_review",
    "build_orchestrator",
    "grade_batch",
    "resolve_exam",
    "resume_batch",
    "sheets_from_transcripts",
    "sqlite_checkpointer",
]


# ── 🗂️ State ────────────────────────────────────────────────────────────────────
def merge_by_student(existing: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer: parallel branches each add {student_id: value}; a later write for the same student wins.

    💡 A dict keyed by student (instead of `operator.add` on a list) makes every step idempotent: if a
       node is re-run after a restart, it overwrites its own entry instead of adding a duplicate.
    """
    return {**(existing or {}), **(new or {})}


class OrchestratorState(TypedDict, total=False):
    """Everything the orchestrator knows, saved after every step.

    💡 Plain JSON-friendly dicts, not Pydantic objects: the SqliteSaver writes this state to disk, and
       plain data is what every checkpointer can store (and what the week 6 API can send to a browser).
    """

    exam_id: str
    exam: dict[str, Any]  # ExamPaper as a dict
    inputs: list[dict[str, Any]]  # [{"student_id", "pdf_path"} or {"student_id", "sheet"}]
    sheets: Annotated[dict[str, dict[str, Any]], merge_by_student]  # student → AnswerSheet
    graded: Annotated[dict[str, dict[str, Any]], merge_by_student]  # student → ReportCard (marks only)
    reports: Annotated[dict[str, dict[str, Any]], merge_by_student]  # student → ReportCard (with feedback)
    files: Annotated[dict[str, dict[str, str]], merge_by_student]  # student → {"html": path, "pdf": path}
    stored: list[str]


def _answered_interrupts() -> int:
    """How many of this node's `interrupt()` calls already have a human's answer. Call it FIRST in the node.

    ⚠️ On resume, LangGraph re-runs the WHOLE node: every `interrupt()` that was already answered
       just returns its answer, and the code before it runs again. So an `emit("...flagged")` placed
       before `interrupt()` would repeat for every student reviewed so far. Comparing a running count
       of interrupts with this number lets us emit only before an interrupt that will really pause.
    💡 It peeks at LangGraph's (private) per-task scratchpad, the same place `interrupt()` itself looks.
       If that ever changes we return 0: a repeated log line, never a lost one.
    """
    try:
        from langgraph.config import get_config

        scratchpad = get_config()["configurable"]["__pregel_scratchpad"]
        # answers from earlier resumes, plus the one just sent with Command(resume=...), if any
        return len(scratchpad.resume) + (scratchpad.get_null_resume(False) is not None)
    except Exception:  # outside a graph, or a LangGraph version with a different scratchpad
        return 0


def _students(state: OrchestratorState) -> list[str]:
    """The students of THIS batch, in the order they were given."""
    return [item["student_id"] for item in state.get("inputs", [])]


# ── 🧑‍🏫 Applying a human's decision (pure functions: easy to test) ──────────────────
def apply_transcription_review(sheet: AnswerSheet, decision: dict[str, Any] | None) -> AnswerSheet:
    """Resume value: {"approved": True, "corrections": {question_id: "corrected text"}}.

    A corrected answer is human-verified, so its confidence becomes 1.0.
    """
    corrections = (decision or {}).get("corrections") or {}
    answers = [
        a.model_copy(update={"text": corrections[a.question_id], "confidence": 1.0, "illegible_spans": []})
        if a.question_id in corrections
        else a
        for a in sheet.answers
    ]
    return sheet.model_copy(update={"answers": answers})


def apply_review(report: ReportCard, decision: dict[str, Any] | None) -> ReportCard:
    """Resume value: {"overrides": {question_id: marks}, "approved": True}.

    - an override replaces that question's mark (clamped to 0…max, principle 7) and is recorded;
    - approved=True clears every flag (the teacher has looked); approved=False keeps them for later.
    """
    decision = decision or {}
    overrides = {str(k): float(v) for k, v in (decision.get("overrides") or {}).items()}
    approved = bool(decision.get("approved", False))
    results = []
    for r in report.results:
        update: dict[str, Any] = {}
        if r.question_id in overrides:
            new = max(0.0, min(overrides[r.question_id], r.max_marks))
            note = f"teacher override {r.awarded_marks:g} → {new:g}"
            update = {"awarded_marks": new, "confidence": 1.0, "review_reason": note}
        if approved and r.needs_review:
            reason = update.get("review_reason") or f"reviewed by teacher (was: {r.review_reason})"
            update.update({"needs_review": False, "review_reason": reason})
        results.append(r.model_copy(update=update) if update else r)
    fresh = ReportCard.from_results(report.student_id, report.exam_id, results)
    return fresh.model_copy(
        update={"strengths": report.strengths, "improvements": report.improvements, "generated_at": report.generated_at}
    )


def resolve_exam(exam_id: str, *, db: DB = None, quiet: bool = False) -> ExamPaper:
    """The approved exam from the database, or, if week 3's approval step was skipped, the exam JSON
    that ships with the course (data/exams/<exam_id>/expected_exam.json)."""
    try:
        return db_load_exam(exam_id, db=db)
    except ExamNotFoundError:
        path = DATA_DIR / "exams" / exam_id / "expected_exam.json"
        if not path.exists():
            raise
        if not quiet:
            tip(f"Exam {exam_id!r} isn't in the database yet; using data/exams/{exam_id}/expected_exam.json.")
        return ExamPaper.model_validate_json(path.read_text(encoding="utf-8"))


# ── 🧱 The graph ─────────────────────────────────────────────────────────────────
def sqlite_checkpointer(path: str | Path | None = None):
    """A SqliteSaver in output/grading_checkpoints.db, so a paused batch survives restarting Python."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(path) if path else CHECKPOINT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    # ⚠️ check_same_thread=False: LangGraph runs parallel branches in worker threads.
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def build_orchestrator(
    checkpointer: BaseCheckpointSaver | Literal["sqlite", "memory"] | None = None,
    llm: Any = None,
    retriever: Any = None,
    *,
    vision_llm: Any = None,
    feedback_llm: Any = None,
    feedback: Literal["llm", "rules"] = "llm",
    transcription_threshold: float | None = None,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    consistency_runs: int = 1,
    use_tools: bool = True,
    db: DB = None,
    index: bool = True,
    embeddings: Any = None,
    persist_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    pdf: bool = True,
):
    """Build and compile the full GradeWise graph.

    Args:
        checkpointer: a saver, "sqlite" (output/grading_checkpoints.db, the default) or "memory".
        llm: the grading model (default grading_model()). Also writes feedback unless feedback_llm is given.
        retriever: evidence retriever for the grader (default: the cs201 knowledge base).
        vision_llm: the transcription model, only used for PDF inputs (default vision_model()).
        feedback: "llm" (strengths/improvements written by a model) or "rules" (free, from the marks).
        transcription_threshold: answers below this confidence pause for review (default: week 4's 0.75).
        review_threshold / consistency_runs / use_tools: passed to the grading graph.
        db / index / embeddings / persist_dir: where reports are stored (SQLite + Chroma reports_<exam>).
        reports_dir / pdf: where the HTML (and PDF) report cards are written (default output/reports/<exam>).
    """
    from gradewise.transcription.confidence import REVIEW_THRESHOLD

    threshold = REVIEW_THRESHOLD if transcription_threshold is None else transcription_threshold
    grading_graph = build_grading_graph(
        llm, retriever, consistency_runs=consistency_runs, review_threshold=review_threshold, use_tools=use_tools
    )
    if feedback == "llm" and feedback_llm is None:
        if llm is not None:
            feedback_llm = llm
        else:
            from gradewise.models import chat_model

            feedback_llm = chat_model()  # 💡 fail fast: a missing key shows up before any grading starts

    def load_exam(state: OrchestratorState) -> dict:
        exam_id = state["exam_id"]
        paper = ExamPaper.model_validate(state["exam"]) if state.get("exam") else resolve_exam(exam_id, db=db)
        emit("load_exam", None, f"{paper.title}: {len(paper.questions)} questions, {len(_students(state))} students")
        return {"exam": paper.model_dump(mode="json")}

    def send_to_transcribe(state: OrchestratorState) -> list[Send]:
        # 💡 One Send per student: every script is transcribed in parallel.
        return [Send("transcribe", {"exam": state["exam"], "student": item}) for item in state["inputs"]]

    def transcribe(task: dict) -> dict:
        student = task["student"]
        sid = student["student_id"]
        if student.get("sheet"):
            sheet = AnswerSheet.model_validate(student["sheet"])
            emit("transcribe", sid, "already transcribed: skipping the vision model", skipped=True)
        else:
            from gradewise.transcription import transcribe_sheet

            emit("transcribe", sid, f"reading {Path(student['pdf_path']).name}…")
            exam = ExamPaper.model_validate(task["exam"])
            sheet = transcribe_sheet(student["pdf_path"], exam, sid, llm=vision_llm, threshold=threshold, quiet=True)
            emit("transcribe", sid, f"{sheet.page_count} pages transcribed", pages=sheet.page_count)
        return {"sheets": {sid: sheet.model_dump(mode="json")}}

    def review_transcription(state: OrchestratorState) -> dict:
        updated = {}
        answered, asked = _answered_interrupts(), 0  # asked = interrupt() calls reached in this run
        for sid in _students(state):
            sheet = AnswerSheet.model_validate(state["sheets"][sid])
            low = [a for a in sheet.answers if a.confidence < threshold]
            if not low:
                continue
            if asked >= answered:  # ⚠️ once per student, not once per resume
                emit("review_transcription", sid, f"{len(low)} low-confidence answer(s) need a human look")
            asked += 1
            # ⏸ Pause. Everything above re-runs on resume, so keep this node free of side effects.
            decision = interrupt(
                {
                    "type": "review_transcription",
                    "student_id": sid,
                    "items": [
                        {
                            "question_id": a.question_id,
                            "text": a.text,
                            "confidence": a.confidence,
                            "illegible_spans": a.illegible_spans,
                            "pages": a.pages,
                        }
                        for a in low
                    ],
                    "resume_with": {"approved": True, "corrections": {"<question_id>": "<corrected text>"}},
                }
            )
            updated[sid] = apply_transcription_review(sheet, decision).model_dump(mode="json")
        return {"sheets": updated} if updated else {}

    def send_to_grade(state: OrchestratorState) -> list[Send]:
        return [Send("grade", {"exam": state["exam"], "sheet": state["sheets"][sid]}) for sid in _students(state)]

    def grade(task: dict) -> dict:
        exam = ExamPaper.model_validate(task["exam"])
        sheet = AnswerSheet.model_validate(task["sheet"])
        emit("grade", sheet.student_id, f"grading {len(exam.questions)} questions…")
        # 💡 The grading graph (Send per question) runs INSIDE this node: a graph inside a graph.
        out = grading_graph.invoke({"exam": exam, "sheet": sheet}, config={"max_concurrency": MAX_PARALLEL_QUESTIONS})
        report: ReportCard = out["report"]
        return {"graded": {sheet.student_id: report.model_dump(mode="json")}}

    def aggregate(state: OrchestratorState) -> dict:
        # 💡 A join node: the Send branches all end here, so the next decision sees EVERY student.
        #    (A conditional edge straight after `grade` would run once per branch, on partial state.)
        reports = [ReportCard.model_validate(state["graded"][sid]) for sid in _students(state)]
        flagged = sum(1 for r in reports for g in r.results if g.needs_review)
        mean = sum(r.percentage for r in reports) / max(1, len(reports))
        emit("aggregate", None, f"{len(reports)} graded · class mean {mean:.1f}% · {flagged} answer(s) flagged")
        return {}

    def after_grading(state: OrchestratorState) -> str:
        flagged = any(ReportCard.model_validate(state["graded"][sid]).needs_review for sid in _students(state))
        return "human_review" if flagged else "write_report"

    def human_review(state: OrchestratorState) -> dict:
        updated = {}
        answered, asked = _answered_interrupts(), 0  # asked = interrupt() calls reached in this run
        for sid in _students(state):
            report = ReportCard.model_validate(state["graded"][sid])
            flagged = [r for r in report.results if r.needs_review]
            if not flagged:
                continue
            # ⚠️ This node re-runs from the top on every resume: without the check, the event log would
            #    say "student_01 flagged" again each time the teacher answers a later student.
            if asked >= answered:
                emit("human_review", sid, f"{len(flagged)} answer(s) flagged for review")
            asked += 1
            # ⏸ Pause until the teacher decides.
            decision = interrupt(
                {
                    "type": "human_review",
                    "student_id": sid,
                    "total": report.total,
                    "max_total": report.max_total,
                    "items": [
                        {
                            "question_id": r.question_id,
                            "awarded": r.awarded_marks,
                            "max_marks": r.max_marks,
                            "reason": r.review_reason,
                            "feedback": r.feedback,
                        }
                        for r in flagged
                    ],
                    "resume_with": {"overrides": {"<question_id>": "<marks>"}, "approved": True},
                }
            )
            updated[sid] = apply_review(report, decision).model_dump(mode="json")
        return {"graded": updated} if updated else {}

    def write_report(state: OrchestratorState) -> dict:
        exam = ExamPaper.model_validate(state["exam"])
        reports, files = {}, {}
        for sid in _students(state):
            report = ReportCard.model_validate(state["graded"][sid])
            if feedback == "llm":
                report = write_feedback(report, exam, llm=feedback_llm)
            else:
                report = rule_based_feedback(report, exam)
            paths = save_report_files(report, exam, reports_dir, pdf=pdf)
            emit(
                "write_report",
                sid,
                f"report card written: {paths['html'].name}",
                **{k: str(v) for k, v in paths.items()},
            )
            reports[sid] = report.model_dump(mode="json")
            files[sid] = {k: str(v) for k, v in paths.items()}
        return {"reports": reports, "files": files}

    def store(state: OrchestratorState) -> dict:
        exam = ExamPaper.model_validate(state["exam"])
        stored = []
        for sid in _students(state):
            report = ReportCard.model_validate(state["reports"][sid])
            store_report(report, exam, db=db, index=index, embeddings=embeddings, persist_dir=persist_dir)
            emit("store", sid, f"saved: {report.total:g}/{report.max_total:g} ({report.grade_band})")
            stored.append(sid)
        emit("done", None, f"{len(stored)} report card(s) stored", students=stored)
        return {"stored": stored}

    graph = StateGraph(OrchestratorState)
    graph.add_node("load_exam", load_exam)
    graph.add_node("transcribe", transcribe)
    graph.add_node("review_transcription", review_transcription)
    graph.add_node("grade", grade)
    graph.add_node("aggregate", aggregate)
    graph.add_node("human_review", human_review)
    graph.add_node("write_report", write_report)
    graph.add_node("store", store)
    graph.add_edge(START, "load_exam")
    graph.add_conditional_edges("load_exam", send_to_transcribe, ["transcribe"])
    graph.add_edge("transcribe", "review_transcription")  # 💡 waits for every student's transcription
    graph.add_conditional_edges("review_transcription", send_to_grade, ["grade"])
    graph.add_edge("grade", "aggregate")  # 💡 waits for every student's grades
    graph.add_conditional_edges("aggregate", after_grading, ["human_review", "write_report"])
    graph.add_edge("human_review", "write_report")
    graph.add_edge("write_report", "store")
    graph.add_edge("store", END)

    if checkpointer is None or checkpointer == "sqlite":
        checkpointer = sqlite_checkpointer()
    elif checkpointer == "memory":
        checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ── 🚀 Running a batch ────────────────────────────────────────────────────────────
@dataclass
class BatchResult:
    """What grade_batch / resume_batch hand back."""

    thread_id: str
    status: Literal["done", "paused"]
    reports: list[ReportCard] = field(default_factory=list)
    files: dict[str, dict[str, str]] = field(default_factory=dict)
    pending: dict[str, Any] | None = None  # the interrupt payload when status == "paused"


TRANSCRIPTS = DATA_DIR / "answer-sheets" / "synthetic" / "transcripts.json"


def sheets_from_transcripts(
    exam: ExamPaper, path: str | Path = TRANSCRIPTS, students: Sequence[str] | int | None = None
) -> list[AnswerSheet]:
    """Build AnswerSheets straight from ground-truth transcripts.json: grading WITHOUT the vision model.

    💡 Perfect transcriptions (confidence 1.0) isolate the grader: if a mark is wrong now, it's the
       grader's fault, not the handwriting's. That's exactly what you want when evaluating grading.
    Args:
        students: None = all, an int = the first N, or a list of student IDs.
    """
    import json

    path = Path(path)
    if not path.exists():
        raise DataNotFoundError(f"Transcripts not found: {path}")
    truth: dict[str, dict[str, str]] = json.loads(path.read_text(encoding="utf-8"))
    ids = list(truth)
    if isinstance(students, int):
        ids = ids[:students]
    elif students is not None:
        ids = [s for s in students if s in truth]
    return [
        AnswerSheet(
            student_id=sid,
            exam_id=exam.exam_id,
            source_file=f"{sid}.pdf",
            page_count=0,
            answers=[
                TranscribedAnswer(question_id=q.id, text=truth[sid].get(q.id, ""), confidence=1.0)
                for q in exam.questions
            ],
        )
        for sid in ids
    ]


def to_inputs(items: Sequence[str | Path | AnswerSheet | dict]) -> list[dict[str, Any]]:
    """PDF paths and/or AnswerSheets → the orchestrator's `inputs` list (student ID = file name)."""
    inputs = []
    for item in items:
        if isinstance(item, AnswerSheet):
            inputs.append({"student_id": item.student_id, "sheet": item.model_dump(mode="json")})
        elif isinstance(item, dict):
            inputs.append(item)
        else:
            path = Path(item)
            if not path.exists():
                raise DataNotFoundError(f"Answer sheet not found: {path}")
            inputs.append({"student_id": path.stem, "pdf_path": str(path)})
    ids = [i["student_id"] for i in inputs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Each student can appear only once in a batch: {ids}")
    return inputs


def _drive(
    app,
    payload: Any,
    config: dict,
    *,
    auto_approve: bool,
    on_event: EventCallback | None,
    reviewer: Reviewer | None,
) -> BatchResult:
    """Stream the graph, relay events, and answer interrupts until it finishes or needs a human."""
    thread_id = config["configurable"]["thread_id"]
    while True:
        for _namespace, mode, chunk in app.stream(payload, config, stream_mode=["updates", "custom"], subgraphs=True):
            if mode == "custom" and on_event and isinstance(chunk, dict) and "stage" in chunk:
                on_event(chunk["stage"], chunk.get("student_id"), chunk.get("detail", {}))
        snapshot = app.get_state(config)
        if not snapshot.interrupts:
            values = snapshot.values
            reports = [ReportCard.model_validate(values["reports"][sid]) for sid in values.get("stored", [])]
            return BatchResult(thread_id, "done", reports, {s: values["files"][s] for s in values.get("stored", [])})
        pending = snapshot.interrupts[0].value
        if on_event:
            on_event("paused", pending.get("student_id"), {"message": f"waiting for {pending['type']}", **pending})
        if auto_approve:
            decision: dict[str, Any] | None = {"approved": True, "overrides": {}, "corrections": {}}
        elif reviewer is not None:
            decision = reviewer(pending)
        else:
            decision = None
        if decision is None:
            return BatchResult(thread_id, "paused", pending=pending)
        if on_event:
            on_event("resumed", pending.get("student_id"), {"message": "decision received", "decision": decision})
        payload = Command(resume=decision)


def grade_batch(
    exam_id: str,
    pdf_paths: Sequence[str | Path | AnswerSheet],
    *,
    thread_id: str,
    auto_approve: bool = False,
    on_event: EventCallback | None = None,
    reviewer: Reviewer | None = None,
    exam: ExamPaper | None = None,
    orchestrator: Any = None,
    **build_kwargs: Any,
) -> BatchResult:
    """Grade a batch of students end to end.

    Args:
        exam_id: which exam (loaded from the database, or data/exams/<id>/expected_exam.json).
        pdf_paths: answer-sheet PDFs and/or already-transcribed AnswerSheets (skips the vision model).
        thread_id: names this batch in the checkpointer. Re-use it to resume a paused batch.
        auto_approve: answer every interrupt with "approved, no changes" (great for demos, not for real marks).
        on_event: progress callback(stage, student_id, detail). See the module docstring.
        reviewer: callback(interrupt_payload) → resume dict, or None to stop and return "paused".
        exam: pass an ExamPaper directly instead of loading it.
        orchestrator: a compiled graph from build_orchestrator(); otherwise one is built from build_kwargs
                      (checkpointer, llm, retriever, vision_llm, feedback, db, embeddings, ...).

    Returns a BatchResult: status "done" with the stored ReportCards, or "paused" with the pending payload.
    💡 If `thread_id` already holds a paused run (say, after a restart), it is resumed instead of restarted.
    """
    app = orchestrator or build_orchestrator(**build_kwargs)
    config = {"configurable": {"thread_id": thread_id}, "max_concurrency": MAX_PARALLEL_STUDENTS}
    if app.get_state(config).next:
        if on_event:
            on_event("resumed", None, {"message": f"thread {thread_id!r} was paused: resuming it"})
        return _drive(app, None, config, auto_approve=auto_approve, on_event=on_event, reviewer=reviewer)
    payload: dict[str, Any] = {"exam_id": exam_id, "inputs": to_inputs(pdf_paths)}
    if exam is not None:
        payload["exam"] = exam.model_dump(mode="json")
    return _drive(app, payload, config, auto_approve=auto_approve, on_event=on_event, reviewer=reviewer)


def resume_batch(
    thread_id: str,
    decision: dict[str, Any],
    *,
    auto_approve: bool = False,
    on_event: EventCallback | None = None,
    reviewer: Reviewer | None = None,
    orchestrator: Any = None,
    **build_kwargs: Any,
) -> BatchResult:
    """Answer the pending interrupt of a paused batch with `decision`, and keep going.

    resume_batch("b1", {"overrides": {"Q7a": 1.5}, "approved": True})
    """
    app = orchestrator or build_orchestrator(**build_kwargs)
    config = {"configurable": {"thread_id": thread_id}, "max_concurrency": MAX_PARALLEL_STUDENTS}
    if not app.get_state(config).interrupts:
        raise ValueError(f"Thread {thread_id!r} is not waiting for a decision.")
    return _drive(
        app, Command(resume=decision), config, auto_approve=auto_approve, on_event=on_event, reviewer=reviewer
    )
