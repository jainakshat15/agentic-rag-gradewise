"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🚢 Week 6 · Day 1: The API's request and response models            │
│                                                                      │
│  What this file does                                                 │
│    Pydantic models for everything that crosses the wire. FastAPI     │
│    uses them three ways at once: to VALIDATE what comes in, to       │
│    SHAPE what goes out, and to write the docs at /docs for free.     │
│                                                                      │
│  💡 The domain contracts (ExamPaper, ReportCard, ...) live in        │
│     gradewise/schemas and are reused here, never redefined.          │
│  ⏭ Used by: main.py, the Streamlit UI (as plain JSON)               │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from gradewise.schemas.exam import ExamPaper

BatchState = Literal["queued", "running", "paused", "done", "failed"]


class ErrorResponse(BaseModel):
    """Every error looks like this: the same four friendly fields as GradeWiseError."""

    error: str = Field(description="A short title, e.g. 'Exam not found'")
    what: str
    cause: str = ""
    fix: str = ""
    help: str = ""


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    demo: bool = Field(description="The server's default mode (each request may override it)")
    provider_key: bool = Field(description="Is a model provider key configured?")
    chat_model: str
    grading_model: str
    vision_model: str
    embedding_model: str = ""
    model_mode: str = Field(default="cheap", description="MODEL_MODE: cheap | free | custom")
    data_size: str = Field(default="full", description="DATA_SIZE: full | small")
    data_banner: str = Field(default="", description="datasets.describe(), e.g. '📦 DATA_SIZE=small: ...'")
    required_keys: list[str] = Field(default_factory=list, description="API-key variables this mode needs")
    jobs: dict[str, int] = Field(default_factory=dict, description="Batches by state")


# ── 📚 Knowledge bases ────────────────────────────────────────────────────────────
class KnowledgeBaseInfo(BaseModel):
    name: str
    files: list[str] = Field(default_factory=list)
    chunks: int = 0
    demo: bool = False


class IngestResponse(KnowledgeBaseInfo):
    added: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    chunks_added: int = 0
    est_cost_usd: float = 0.0


# ── 📝 Exams ─────────────────────────────────────────────────────────────────────
class ExamReview(BaseModel):
    """The exam graph's state after upload: usually paused, waiting for Emma's approval."""

    exam_id: str
    thread_id: str
    status: Literal["awaiting_approval", "stored", "rejected"]
    message: str = ""
    summary: dict[str, Any] | None = None  # gradewise.exam.summary.exam_summary()
    problems: list[str] = Field(default_factory=list)
    paper: dict[str, Any] | None = None  # the extracted ExamPaper, as JSON
    demo: bool = False


class ApproveExamRequest(BaseModel):
    approved: bool = True
    edits: dict[str, Any] | None = Field(default=None, description="Fields of the ExamPaper to replace")
    demo: bool | None = None


class ExamListItem(BaseModel):
    exam_id: str
    title: str
    subject: str
    total_marks: float
    question_count: int
    source: Literal["database", "sample"] = "database"
    updated_at: datetime | None = None


class ExamDetail(BaseModel):
    source: Literal["database", "sample"]
    paper: ExamPaper
    summary: dict[str, Any]


# ── 🧮 Batches ───────────────────────────────────────────────────────────────────
class BatchCreated(BaseModel):
    batch_id: str
    status: BatchState
    students: list[str]
    events_url: str


class ReviewCriterion(BaseModel):
    criterion_id: str
    awarded: float
    max_marks: float | None = None
    description: str = ""
    justification: str = ""
    evidence: list[str] = Field(default_factory=list)


class ReviewItem(BaseModel):
    """One flagged answer, with everything a teacher needs to decide on it."""

    question_id: str
    question_text: str = ""
    max_marks: float | None = None
    awarded: float | None = None  # human_review only
    reason: str | None = None
    feedback: str | None = None
    answer_text: str = Field(default="", description="What the student wrote (the transcription)")
    confidence: float | None = None
    illegible_spans: list[str] = Field(default_factory=list)
    criteria: list[ReviewCriterion] = Field(default_factory=list)


class PendingReview(BaseModel):
    type: Literal["human_review", "review_transcription"]
    student_id: str
    total: float | None = None
    max_total: float | None = None
    items: list[ReviewItem]
    resume_with: dict[str, Any] = Field(default_factory=dict)


class StudentProgress(BaseModel):
    student_id: str
    stage: str = "queued"
    questions_done: int = 0
    questions_total: int = 0
    percent: int = 0


class ReportSummary(BaseModel):
    student_id: str
    total: float
    max_total: float
    percentage: float
    grade_band: str
    needs_review: bool = False


class BatchStatus(BaseModel):
    batch_id: str
    exam_id: str
    status: BatchState
    demo: bool
    created_at: datetime
    students: list[str]
    progress: list[StudentProgress] = Field(default_factory=list)
    pending: PendingReview | None = None
    reports: list[ReportSummary] = Field(default_factory=list)
    files: dict[str, dict[str, str]] = Field(default_factory=dict)
    events: int = Field(0, description="How many progress events so far (stream them from /events)")
    error: ErrorResponse | None = None


class ReviewDecision(BaseModel):
    """Emma's decision on the pending review. Send only what applies to the pending type."""

    student_id: str | None = Field(default=None, description="Optional safety check: must match the pending one")
    overrides: dict[str, float] = Field(default_factory=dict, description="human_review: {question_id: marks}")
    corrections: dict[str, str] = Field(default_factory=dict, description="review_transcription: {qid: text}")
    approved: bool = True


# ── 📊 Reports & chat ───────────────────────────────────────────────────────────
class QuestionStat(BaseModel):
    question_id: str
    max_marks: float
    mean: float
    mean_percent: float
    flagged: int = 0


class ClassReport(BaseModel):
    exam_id: str
    count: int
    mean_percent: float = 0.0
    median_percent: float = 0.0
    bands: dict[str, int] = Field(default_factory=dict)
    flagged: int = 0
    students: list[ReportSummary] = Field(default_factory=list)
    questions: list[QuestionStat] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    exam_id: str = "cs_midterm"
    demo: bool | None = None


class AskResponse(BaseModel):
    answer: str
    exam_id: str
    demo: bool
