"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🚢 Week 6 · Day 1: Background batches and their progress            │
│                                                                      │
│  What this file does                                                 │
│    Grading a class takes minutes; an HTTP request should take        │
│    milliseconds. So POST /batches only STARTS a job and hands back   │
│    a batch_id. The job runs grade_batch() in a thread pool, records  │
│    every on_event() as a numbered event (the SSE stream replays      │
│    them), and stops at each interrupt until Emma reviews.            │
│                                                                      │
│      BatchJob      → one batch: status, events, progress, pending    │
│      JobRegistry   → the in-memory dict of jobs + the thread pool    │
│                                                                      │
│  ⏮ Built on: gradewise/orchestrator.py (grade_batch, resume_batch)  │
│  ⏭ Used by: main.py                                                 │
╰──────────────────────────────────────────────────────────────────────╯

🚀 PRO: IN PRODUCTION, THE JOB REGISTRY MOVES OUT OF THE WEB PROCESS
   This registry is a Python dict inside the API process. That's perfect for a course and a single
   server, but it has three limits: restart the API and the job list is gone (the LangGraph checkpoints
   in SQLite survive, so the WORK is not lost); you can't run two API replicas; and a crash in grading
   shares a process with the web server. The usual upgrades, in order of effort:
     1. A task queue: Celery, RQ, Dramatiq or Arq, with Redis (or RabbitMQ) as the broker. The API
        enqueues "grade batch b-123"; separate worker processes pick it up.
     2. Progress via Redis: workers XADD events to a Redis Stream per batch; the SSE endpoint XREADs
        them (with the last event ID, so a reconnecting browser misses nothing). Job status lives in
        a Redis hash or a `batches` table in Postgres instead of this dict.
     3. Checkpoints in Postgres (langgraph-checkpoint-postgres) so any worker can resume any batch.
     4. Or let LangGraph Platform host the graph: it ships runs, threads, interrupts and streaming
        as a managed API (see week6/project/DEPLOY.md).
"""

from __future__ import annotations

import threading
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fastapi.encoders import jsonable_encoder

from gradewise.api.workspace import Workspace
from gradewise.console import warn
from gradewise.errors import GradeWiseError
from gradewise.orchestrator import BatchResult
from gradewise.schemas.exam import ExamPaper

MAX_WORKERS = 2  # 💡 two batches grade at once; each one already runs students and questions in parallel
STAGE_PERCENT = {
    "queued": 0,
    "transcribe": 15,
    "review_transcription": 15,
    "grade": 20,
    "aggregate": 80,
    "human_review": 85,
    "write_report": 92,
    "store": 100,
}


def error_payload(err: BaseException) -> dict[str, str]:
    """Any exception → the four friendly fields. GradeWiseErrors already carry them."""
    if isinstance(err, GradeWiseError):
        return {"error": err.title, "what": err.what, "cause": err.cause, "fix": err.fix, "help": err.help_ref}
    return {
        "error": "Unexpected error",
        "what": f"{type(err).__name__}: {err}",
        "cause": "Something GradeWise didn't anticipate went wrong.",
        "fix": "Check the API's terminal for the full traceback.",
        "help": "setup/troubleshooting.ipynb",
    }


def new_batch_id() -> str:
    return f"b-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


@dataclass
class BatchJob:
    """One batch of answer sheets on its way to report cards."""

    batch_id: str
    exam: ExamPaper
    students: list[str]
    workspace: Workspace
    orchestrator: Any
    auto_approve: bool = False
    status: str = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    events: list[dict[str, Any]] = field(default_factory=list)
    progress: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending: dict[str, Any] | None = None
    result: BatchResult | None = None
    error: dict[str, str] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        total = len(self.exam.questions)
        for sid in self.students:
            self.progress[sid] = {"student_id": sid, "stage": "queued", "questions_done": 0, "questions_total": total}
            self.progress[sid]["percent"] = 0

    @property
    def thread_id(self) -> str:
        return self.batch_id  # 💡 one LangGraph thread per batch: the checkpointer's key

    @property
    def demo(self) -> bool:
        return self.workspace.demo

    @property
    def config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id}}

    # ── 📡 events ──────────────────────────────────────────────────────────────
    def on_event(self, stage: str, student_id: str | None, detail: dict[str, Any]) -> None:
        """The on_event callback handed to grade_batch / resume_batch. Called from a worker thread."""
        with self.lock:
            event = {
                "id": len(self.events) + 1,
                "time": datetime.now(UTC).isoformat(),
                "stage": stage,
                "student_id": student_id,
                "message": detail.get("message", ""),
                "detail": jsonable_encoder({k: v for k, v in detail.items() if k != "message"}),
            }
            self._update_progress(stage, student_id)
            if student_id in self.progress:
                event["progress"] = dict(self.progress[student_id])
            self.events.append(event)

    def _update_progress(self, stage: str, student_id: str | None) -> None:
        if stage == "done":
            for p in self.progress.values():
                p.update(stage="done", percent=100)
            return
        p = self.progress.get(student_id or "")
        if p is None:
            return
        if stage == "grade_question":
            p["questions_done"] = min(p["questions_done"] + 1, p["questions_total"])
            share = p["questions_done"] / max(1, p["questions_total"])
            p["percent"] = max(p["percent"], 20 + int(60 * share))
        else:
            p["percent"] = max(p["percent"], STAGE_PERCENT.get(stage, p["percent"]))
        if stage not in ("paused", "resumed"):
            p["stage"] = stage

    def events_after(self, last_id: int) -> list[dict[str, Any]]:
        with self.lock:
            return self.events[last_id:]

    # ── 🏁 results ─────────────────────────────────────────────────────────────
    def finish(self, result: BatchResult) -> None:
        with self.lock:
            self.result = result
            self.pending = result.pending
            self.status = result.status
            if result.status == "paused" and result.pending:
                sid = result.pending.get("student_id")
                if sid in self.progress:
                    self.progress[sid]["stage"] = result.pending["type"]

    def fail(self, err: BaseException) -> None:
        with self.lock:
            self.error = error_payload(err)
            self.status = "failed"
            self.pending = None
        self.on_event("failed", None, {"message": self.error["what"], **self.error})

    def state_values(self) -> dict[str, Any]:
        """The orchestrator's saved state for this batch (sheets, grades, ...), straight from the checkpointer."""
        try:
            return self.orchestrator.get_state(self.config).values or {}
        except Exception:  # noqa: BLE001 - a missing checkpoint just means "nothing to show yet"
            return {}


class JobRegistry:
    """Every batch this API process knows about, plus the pool of worker threads that runs them."""

    def __init__(self, max_workers: int = MAX_WORKERS) -> None:
        self.jobs: dict[str, BatchJob] = {}
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gradewise-batch")
        self._lock = threading.Lock()

    def add(self, job: BatchJob) -> BatchJob:
        with self._lock:
            self.jobs[job.batch_id] = job
        return job

    def get(self, batch_id: str) -> BatchJob | None:
        return self.jobs.get(batch_id)

    def all(self) -> list[BatchJob]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for job in self.jobs.values():
            counts[job.status] = counts.get(job.status, 0) + 1
        return counts

    def submit(self, job: BatchJob, work: Callable[[], BatchResult]) -> None:
        """Run `work` (a grade_batch / resume_batch call) in the pool and record how it ends."""
        with job.lock:
            job.status = "running"
            job.pending = None

        def run() -> None:
            try:
                job.finish(work())
            except Exception as exc:  # noqa: BLE001 - a failed batch must never take the API down
                warn(f"Batch {job.batch_id} failed: {exc}")
                traceback.print_exc()
                job.fail(exc)

        self.pool.submit(run)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
