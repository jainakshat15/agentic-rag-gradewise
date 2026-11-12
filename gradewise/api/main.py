"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🚢 Week 6 · Day 1: The GradeWise API (FastAPI)                      │
│                                                                      │
│  What this file does                                                 │
│    Every GradeWise capability, as an HTTP endpoint:                  │
│                                                                      │
│      GET  /health                          is it up? which mode?     │
│      GET  /knowledge-bases                 what's been ingested      │
│      POST /knowledge-bases/{name}/documents upload PDFs → ingest     │
│      POST /exams                           upload exam pack → ⏸      │
│      POST /exams/{id}/approve              Emma approves → stored    │
│      GET  /exams · /exams/{id}             stored exams              │
│      POST /batches                         start grading (background)│
│      GET  /batches · /batches/{id}         status + pending review   │
│      GET  /batches/{id}/events             live progress (SSE)       │
│      POST /batches/{id}/review             overrides + approve → ▶  │
│      GET  /reports/{exam} · /{exam}/{sid}  JSON, ?format=html|pdf    │
│      POST /ask                             chat with the results     │
│                                                                      │
│  Run : uv run python -m gradewise.api      → http://localhost:8000   │
│        GRADEWISE_DEMO=true … for the free, offline demo mode         │
│  ⏮ Built on: orchestrator.py, exam/graph.py, reports/, ingestion/   │
│  ⏭ Used by: gradewise/ui (Streamlit), docker/, tests/test_api.py    │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import asyncio
import json
import shutil
import statistics
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from langgraph.types import Command

from gradewise import __version__, datasets
from gradewise.api.demo import DemoExamModel, DemoGraderModel, DemoResultsModel, DemoRetriever, demo_embeddings
from gradewise.api.demo import demo_sheet as make_demo_sheet
from gradewise.api.jobs import BatchJob, JobRegistry, error_payload, new_batch_id
from gradewise.api.schemas import (
    ApproveExamRequest,
    AskRequest,
    AskResponse,
    BatchCreated,
    BatchStatus,
    ClassReport,
    ErrorResponse,
    ExamDetail,
    ExamListItem,
    ExamReview,
    Health,
    IngestResponse,
    KnowledgeBaseInfo,
    PendingReview,
    QuestionStat,
    ReportSummary,
    ReviewCriterion,
    ReviewDecision,
    ReviewItem,
    StudentProgress,
)
from gradewise.api.workspace import (
    InvalidNameError,
    Workspace,
    demo_from_env,
    has_provider_key,
    make_workspaces,
    safe_filename,
    safe_name,
)
from gradewise.config import DATA_DIR, settings
from gradewise.db import ExamNotFoundError, ReportNotFoundError, get_report, list_exams, list_reports
from gradewise.db import load_exam as db_load_exam
from gradewise.errors import DataNotFoundError, GradeWiseError, InvalidModelError, MissingAPIKeyError
from gradewise.exam.summary import exam_summary
from gradewise.schemas.exam import ExamPaper
from gradewise.schemas.report import ReportCard

COURSE_NOTES = DATA_DIR / "knowledge-base"
SYNTHETIC_SHEETS = DATA_DIR / "answer-sheets" / "synthetic"
SAMPLE_EXAM_ID = "cs_midterm"
TERMINAL = ("paused", "done", "failed")


# ── ❌ Friendly errors ───────────────────────────────────────────────────────────
class BatchNotFoundError(GradeWiseError):
    title = "Batch not found"
    cause = "This API process has no batch with that ID. Jobs live in memory, so a restart forgets them."
    fix = "List the batches with GET /batches, or start a new one with POST /batches."
    help_ref = "week6/project/README.md"


class NotWaitingError(GradeWiseError):
    title = "Nothing is waiting for a decision"
    cause = "The batch (or exam) is still running, already finished, or was reviewed a moment ago."
    fix = "Check GET /batches/{id} (or /exams/{id}) and only review when status is 'paused'."
    help_ref = "week6/project/README.md"


class BadUploadError(GradeWiseError):
    title = "That upload can't be used"
    cause = "A required file is missing, or a file isn't really a PDF."
    fix = "Upload real .pdf files (exported from a scanner or Word), one per field."
    help_ref = "week6/project/README.md"


# 💡 A GradeWiseError subclass → an HTTP status. The body always carries what / cause / fix / help.
STATUS_CODES: list[tuple[type[GradeWiseError], int]] = [
    (ExamNotFoundError, 404),
    (ReportNotFoundError, 404),
    (BatchNotFoundError, 404),
    (DataNotFoundError, 404),
    (NotWaitingError, 409),
    (InvalidNameError, 422),
    (BadUploadError, 400),
    (MissingAPIKeyError, 503),
    (InvalidModelError, 503),
]


def status_for(err: GradeWiseError) -> int:
    if type(err).__name__ in ("NoReportsError", "EmptyKnowledgeBaseError"):
        return 404
    if type(err).__name__ == "ExamExtractionError":
        return 422
    return next((code for cls, code in STATUS_CODES if isinstance(err, cls)), 400)


# ── 🧰 Small helpers ─────────────────────────────────────────────────────────────
def workspace(request: Request, demo: bool | None) -> Workspace:
    """The request's workspace: its own `demo` flag if given, otherwise the server default."""
    use_demo = request.app.state.default_demo if demo is None else demo
    return request.app.state.workspaces[bool(use_demo)]


def jobs(request: Request) -> JobRegistry:
    return request.app.state.jobs


def save_upload(upload: UploadFile, folder: Path, name: str | None = None, *, pdf: bool = True) -> Path:
    """Stream an uploaded file to disk. ⚠️ Check it's what it claims to be: the browser's word isn't proof.

    🚀 PRO: cap the size too (a reverse proxy like nginx's client_max_body_size, or count bytes here).
    """
    filename = safe_filename(name or upload.filename)
    head = upload.file.read(5)
    upload.file.seek(0)
    if pdf and head != b"%PDF-":
        raise BadUploadError(f"{upload.filename!r} is not a PDF file.")
    path = folder / filename
    with path.open("wb") as out:
        shutil.copyfileobj(upload.file, out)
    return path


def resolve_paper(exam_id: str, ws: Workspace) -> tuple[ExamPaper, Literal["database", "sample"]]:
    """An approved exam from this workspace's database, else the JSON that ships with the course."""
    from gradewise.orchestrator import resolve_exam

    safe_name(exam_id, "exam ID")
    try:
        return db_load_exam(exam_id, db=ws.db), "database"
    except ExamNotFoundError:
        return resolve_exam(exam_id, db=ws.db, quiet=True), "sample"


def summarise(report: ReportCard) -> ReportSummary:
    return ReportSummary(
        student_id=report.student_id,
        total=report.total,
        max_total=report.max_total,
        percentage=report.percentage,
        grade_band=report.grade_band,
        needs_review=report.needs_review,
    )


def vectorstore_counts(ws: Workspace) -> dict[str, int]:
    """Chunks per knowledge-base collection in this workspace (report collections left out)."""
    from gradewise.ingestion.store import get_vectorstore

    if not ws.vector_dir.exists():
        return {}
    # 💡 Counting needs no embeddings, so the free fake works in both modes (no API key involved).
    #    The store is keyed by embedding model: demo mode has its own folder, real mode uses your
    #    MODEL_MODE's embedding model, so we open THAT folder (embedding_model=...).
    key = None if ws.demo else settings.embedding_model
    client = get_vectorstore(
        "cs201", embeddings=demo_embeddings(), persist_dir=ws.vector_dir, embedding_model=key
    )._client
    counts = {}
    for col in client.list_collections():
        name = col if isinstance(col, str) else col.name
        if not name.startswith("reports_"):
            counts[name] = client.get_collection(name).count()
    return counts


def pick_students(count: int | None) -> list[str]:
    """The synthetic students for a batch: datasets.synthetic_students() (10, or 2 when DATA_SIZE=small),
    cut to the first `count`. Asking for more than the small set falls back to the full class."""
    ids = datasets.synthetic_students()
    if count is not None and count > len(ids):
        ids = datasets.synthetic_students(small=False)
    return ids[:count] if count else ids


def pick_questions(paper: ExamPaper, requested: str | None) -> list[str]:
    """The question IDs to grade: the comma-separated `requested`, else datasets.exam_question_ids(paper)."""
    if requested and requested.strip():
        ids = [q.strip() for q in requested.split(",") if q.strip()]
        unknown = sorted(set(ids) - {q.id for q in paper.questions})
        if unknown:
            raise BadUploadError(
                f"These question IDs are not on exam {paper.exam_id!r}: {unknown}",
                fix=f"Use IDs from the exam, e.g. {', '.join(q.id for q in paper.questions[:4])}.",
            )
        return ids
    return datasets.exam_question_ids(paper)


# ── 🧮 Building the graphs for a workspace ──────────────────────────────────────
def exam_graph(request: Request, ws: Workspace):
    """One compiled exam graph per workspace, reused, with a SQLite checkpointer so approvals survive restarts."""
    cache: dict[bool, Any] = request.app.state.exam_graphs
    if ws.demo not in cache:
        from gradewise.exam.graph import build_exam_graph, sqlite_checkpointer

        ws.root.mkdir(parents=True, exist_ok=True)
        llm = DemoExamModel() if ws.demo else None  # 💡 None → chat_model(), created on first use
        cache[ws.demo] = build_exam_graph(sqlite_checkpointer(ws.exam_checkpoints), llm=llm, db=ws.db, write_json=False)
    return cache[ws.demo]


def batch_orchestrator(ws: Workspace, exam: ExamPaper):
    """The week 5 orchestrator, wired to this workspace (and to the fakes in demo mode)."""
    from gradewise.orchestrator import build_orchestrator, sqlite_checkpointer

    ws.root.mkdir(parents=True, exist_ok=True)
    common: dict[str, Any] = {
        "checkpointer": sqlite_checkpointer(ws.batch_checkpoints),
        "db": ws.db,
        "persist_dir": ws.vector_dir,
        "reports_dir": ws.reports_dir(exam.exam_id),
    }
    if ws.demo:
        retriever = DemoRetriever(collection=exam.knowledge_base, persist_dir=str(ws.vector_dir))
        return build_orchestrator(
            llm=DemoGraderModel(), retriever=retriever, feedback="rules", embeddings=demo_embeddings(), **common
        )
    from gradewise.rag import get_retriever

    # 💡 Built HERE, inside the request, so a missing API key is a friendly 503 now, not a failed job later.
    retriever = get_retriever(collection=exam.knowledge_base, persist_dir=ws.vector_dir)
    return build_orchestrator(retriever=retriever, **common)


def exam_review(graph, exam_id: str, thread_id: str, demo: bool) -> ExamReview:
    """Turn the exam graph's saved state into a response: paused for approval, stored, or rejected."""
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    if snapshot.interrupts:
        payload = snapshot.interrupts[0].value
        return ExamReview(
            exam_id=exam_id,
            thread_id=thread_id,
            status="awaiting_approval",
            message=payload.get("message", ""),
            summary=payload.get("summary"),
            problems=payload.get("problems", []),
            paper=payload.get("paper"),
            demo=demo,
        )
    values = snapshot.values or {}
    paper = values.get("paper")
    return ExamReview(
        exam_id=exam_id,
        thread_id=thread_id,
        status="stored" if values.get("stored") else "rejected",
        message="Approved and stored." if values.get("stored") else "Not approved: nothing was stored.",
        summary=exam_summary(ExamPaper.model_validate(paper)) if paper else None,
        problems=values.get("problems", []),
        paper=paper,
        demo=demo,
    )


def pending_review(job: BatchJob) -> PendingReview | None:
    """The interrupt payload, enriched with what Emma needs: question, transcription, justification, evidence."""
    pending = job.pending
    if not pending:
        return None
    values = job.state_values()
    sid = pending["student_id"]
    questions = {q.id: q for q in job.exam.questions}
    sheet = (values.get("sheets") or {}).get(sid) or {}
    answers = {a["question_id"]: a for a in sheet.get("answers", [])}
    graded = (values.get("graded") or {}).get(sid) or {}
    results = {r["question_id"]: r for r in graded.get("results", [])}

    items = []
    for item in pending.get("items", []):
        qid = item["question_id"]
        question, answer, result = questions.get(qid), answers.get(qid, {}), results.get(qid, {})
        rubric = {c.id: c for c in question.criteria} if question else {}
        criteria = [
            ReviewCriterion(
                criterion_id=c["criterion_id"],
                awarded=c["awarded"],
                max_marks=rubric[c["criterion_id"]].marks if c["criterion_id"] in rubric else None,
                description=rubric[c["criterion_id"]].description if c["criterion_id"] in rubric else "",
                justification=c.get("justification", ""),
                evidence=c.get("evidence", []),
            )
            for c in result.get("criteria", [])
        ]
        items.append(
            ReviewItem(
                question_id=qid,
                question_text=question.text if question else "",
                max_marks=item.get("max_marks") or (question.max_marks if question else None),
                awarded=item.get("awarded"),
                reason=item.get("reason"),
                feedback=item.get("feedback"),
                answer_text=item.get("text", answer.get("text", "")),
                confidence=item.get("confidence", answer.get("confidence")),
                illegible_spans=item.get("illegible_spans", answer.get("illegible_spans", [])),
                criteria=criteria,
            )
        )
    return PendingReview(
        type=pending["type"],
        student_id=sid,
        total=pending.get("total"),
        max_total=pending.get("max_total"),
        items=items,
        resume_with=pending.get("resume_with", {}),
    )


def batch_status(job: BatchJob) -> BatchStatus:
    result = job.result
    reports = [summarise(r) for r in result.reports] if result and result.status == "done" else []
    return BatchStatus(
        batch_id=job.batch_id,
        exam_id=job.exam.exam_id,
        status=job.status,
        demo=job.demo,
        created_at=job.created_at,
        students=job.students,
        progress=[StudentProgress(**p) for p in job.progress.values()],
        pending=pending_review(job) if job.status == "paused" else None,
        reports=reports,
        files=result.files if result and result.status == "done" else {},
        events=len(job.events),
        error=ErrorResponse(**job.error) if job.error else None,
    )


def get_job(request: Request, batch_id: str) -> BatchJob:
    job = jobs(request).get(batch_id)
    if job is None:
        raise BatchNotFoundError(f"There is no batch {batch_id!r}.")
    return job


# ── 🛣️ Routes ───────────────────────────────────────────────────────────────────
# 💡 Plain `def` routes (not `async def`) for anything that does real work: FastAPI runs them in a
#    thread pool, so a slow ingest or LLM call never blocks the event loop that serves everyone else.
router = APIRouter()
DemoQuery = Annotated[bool | None, Query(description="true = offline fakes (default: GRADEWISE_DEMO)")]
DemoForm = Annotated[bool | None, Form(description="true = offline fakes (default: GRADEWISE_DEMO)")]


@router.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"name": "GradeWise API", "docs": "/docs", "health": "/health"}


@router.get("/health", response_model=Health, tags=["system"])
def health(request: Request) -> Health:
    """Is the API up, which mode is the default, and which models would it use?"""
    return Health(
        version=__version__,
        demo=request.app.state.default_demo,
        provider_key=has_provider_key(),
        chat_model=settings.chat_model,
        grading_model=settings.grading_model,
        vision_model=settings.transcription_model,
        embedding_model=settings.embedding_model,
        model_mode=settings.model_mode,
        data_size=datasets.data_size(),
        data_banner=datasets.describe(),
        required_keys=settings.required_keys,
        jobs=jobs(request).counts(),
    )


# 📚 Knowledge bases ────────────────────────────────────────────────────────────
@router.get("/knowledge-bases", response_model=list[KnowledgeBaseInfo], tags=["knowledge bases"])
def list_knowledge_bases(request: Request, demo: DemoQuery = None) -> list[KnowledgeBaseInfo]:
    ws = workspace(request, demo)
    counts = vectorstore_counts(ws)
    kb_root = ws.upload_dir("kb")
    names = sorted(set(counts) | {p.name for p in kb_root.iterdir() if p.is_dir()})
    return [
        KnowledgeBaseInfo(
            name=name,
            files=sorted(p.name for p in (kb_root / name).glob("*.pdf")) if (kb_root / name).exists() else [],
            chunks=counts.get(name, 0),
            demo=ws.demo,
        )
        for name in names
    ]


@router.post(
    "/knowledge-bases/{name}/documents", response_model=IngestResponse, status_code=201, tags=["knowledge bases"]
)
def upload_documents(
    request: Request,
    name: str,
    files: Annotated[list[UploadFile], File(description="Course-note PDFs")] = [],  # noqa: B006
    include_course_notes: Annotated[
        bool, Form(description="Also add the CS201 notes from data/ (2 of the 4 PDFs when DATA_SIZE=small)")
    ] = False,
    reset: Annotated[bool, Form(description="Wipe the collection first")] = False,
    demo: DemoForm = None,
) -> IngestResponse:
    """Upload PDFs into output/uploads/…/kb/<name>/ and ingest that folder (unchanged files are skipped)."""
    from gradewise.ingestion import ingest

    ws = workspace(request, demo)
    folder = ws.upload_dir("kb", safe_name(name, "knowledge-base name"))
    for upload in files:
        save_upload(upload, folder)
    if include_course_notes:
        for pdf in datasets.kb_files(COURSE_NOTES):
            shutil.copy2(pdf, folder / pdf.name)  # 💡 a COPY: data/ is never written to
    if not any(folder.glob("*.pdf")):
        raise BadUploadError(f"Knowledge base {name!r} has no PDFs yet.", fix="Upload at least one PDF.")
    report = ingest(
        folder,
        collection=name,
        embeddings=demo_embeddings() if ws.demo else None,
        reset=reset,
        persist_dir=ws.vector_dir,
        quiet=True,
    )
    return IngestResponse(
        name=name,
        files=sorted(p.name for p in folder.glob("*.pdf")),
        chunks=report.chunks_total,
        demo=ws.demo,
        added=report.added,
        updated=report.updated,
        skipped=report.skipped,
        chunks_added=report.chunks_added,
        est_cost_usd=0.0 if ws.demo else report.est_cost_usd,
    )


# 📝 Exams ──────────────────────────────────────────────────────────────────────
@router.post("/exams", response_model=ExamReview, status_code=201, tags=["exams"])
def upload_exam(
    request: Request,
    exam_id: Annotated[str, Form()] = SAMPLE_EXAM_ID,
    question_paper: Annotated[UploadFile | None, File()] = None,
    marking_scheme: Annotated[UploadFile | None, File()] = None,
    methodology: Annotated[UploadFile | None, File()] = None,
    model_answers: Annotated[UploadFile | None, File(description="Optional .md or .pdf")] = None,
    use_sample: Annotated[bool, Form(description="Use the bundled CS201 midterm pack instead")] = False,
    demo: DemoForm = None,
) -> ExamReview:
    """Upload an exam pack, run the week 3 exam graph, and return the interrupt: the exam awaiting approval."""
    from gradewise.exam.extract import exam_pack
    from gradewise.exam.graph import start_input

    ws = workspace(request, demo)
    safe_name(exam_id, "exam ID")
    if use_sample:
        sources = {key: str(path) for key, path in exam_pack(SAMPLE_EXAM_ID).items()}
    else:
        given = {"question_paper": question_paper, "marking_scheme": marking_scheme, "methodology": methodology}
        missing = [key for key, upload in given.items() if upload is None]
        if missing:
            raise BadUploadError(f"Missing file(s): {', '.join(missing)}.", fix="Upload all three PDFs.")
        folder = ws.upload_dir("exams", exam_id)
        sources = {key: str(save_upload(upload, folder, f"{key}.pdf")) for key, upload in given.items()}
        if model_answers is not None:
            suffix = Path(model_answers.filename or "x.md").suffix.lower()
            is_pdf = suffix == ".pdf"
            path = save_upload(model_answers, folder, f"model_answers{'.pdf' if is_pdf else '.md'}", pdf=is_pdf)
            sources["model_answers"] = str(path)

    graph = exam_graph(request, ws)
    thread_id = f"exam-{exam_id}"  # 💡 deterministic: approve still finds it after an API restart
    # 🚀 PRO: real extraction takes ~30 s. Fine for an admin page; for more, make this a background job too.
    graph.invoke(start_input(exam_id, sources=sources), {"configurable": {"thread_id": thread_id}})
    return exam_review(graph, exam_id, thread_id, ws.demo)


@router.post("/exams/{exam_id}/approve", response_model=ExamReview, tags=["exams"])
def approve_exam(request: Request, exam_id: str, body: ApproveExamRequest) -> ExamReview:
    """Resume the paused exam graph with Emma's decision: approve (optionally with edits) or reject."""
    ws = workspace(request, body.demo)
    graph = exam_graph(request, ws)
    thread_id = f"exam-{safe_name(exam_id, 'exam ID')}"
    config = {"configurable": {"thread_id": thread_id}}
    if not graph.get_state(config).interrupts:
        raise NotWaitingError(f"Exam {exam_id!r} is not waiting for approval.")
    graph.invoke(Command(resume={"approved": body.approved, "edits": body.edits}), config)
    return exam_review(graph, exam_id, thread_id, ws.demo)


@router.get("/exams", response_model=list[ExamListItem], tags=["exams"])
def get_exams(request: Request, demo: DemoQuery = None) -> list[ExamListItem]:
    """Every approved exam, plus the bundled sample exam if it hasn't been approved here yet."""
    ws = workspace(request, demo)
    items = [
        ExamListItem(
            exam_id=r.exam_id,
            title=r.title,
            subject=r.subject,
            total_marks=r.total_marks,
            question_count=r.question_count,
            updated_at=r.updated_at,
        )
        for r in list_exams(db=ws.db)
    ]
    if SAMPLE_EXAM_ID not in {i.exam_id for i in items}:
        try:
            paper, source = resolve_paper(SAMPLE_EXAM_ID, ws)
            items.append(
                ExamListItem(
                    exam_id=paper.exam_id,
                    title=paper.title,
                    subject=paper.subject,
                    total_marks=paper.total_marks,
                    question_count=len(paper.questions),
                    source=source,
                )
            )
        except GradeWiseError:
            pass
    return items


@router.get("/exams/{exam_id}", response_model=ExamDetail, tags=["exams"])
def get_exam(request: Request, exam_id: str, demo: DemoQuery = None) -> ExamDetail:
    paper, source = resolve_paper(exam_id, workspace(request, demo))
    return ExamDetail(source=source, paper=paper, summary=exam_summary(paper))


# 🧮 Batches ────────────────────────────────────────────────────────────────────
@router.post("/batches", response_model=BatchCreated, status_code=202, tags=["batches"])
def create_batch(
    request: Request,
    exam_id: Annotated[str, Form()] = SAMPLE_EXAM_ID,
    files: Annotated[list[UploadFile], File(description="Answer-sheet PDFs (student ID = file name)")] = [],  # noqa: B006
    use_synthetic: Annotated[bool, Form(description="Grade the bundled synthetic answer PDFs")] = False,
    from_transcripts: Annotated[bool, Form(description="Skip vision: grade transcripts.json")] = False,
    students: Annotated[
        int | None,
        Form(ge=1, le=100, description="How many synthetic students (default: 10, or 2 when DATA_SIZE=small)"),
    ] = None,
    auto_approve: Annotated[bool, Form(description="Approve every review automatically (demos only!)")] = False,
    questions: Annotated[
        str | None,
        Form(
            description="Comma-separated question IDs to grade, e.g. Q1,Q2a (default: all, or Q1–Q3 if DATA_SIZE=small)"
        ),
    ] = None,
    demo: DemoForm = None,
) -> BatchCreated:
    """Start grading in the background. Returns at once with a batch_id: poll it, or stream its /events."""
    from gradewise.orchestrator import grade_batch, sheets_from_transcripts

    ws = workspace(request, demo)
    paper, _ = resolve_paper(exam_id, ws)
    # 📦 DATA_SIZE=small grades Q1–Q3 only; an explicit `questions` form field always wins.
    question_ids = pick_questions(paper, questions)
    paper = datasets.subset_exam(paper, question_ids)
    batch_id = new_batch_id()
    items: list[Any]
    chosen = pick_students(students)
    if from_transcripts:
        items = sheets_from_transcripts(paper, students=chosen)
        how = f"{len(items)} answer sheet(s) from transcripts.json (no vision model)"
    else:
        if files:
            folder = ws.upload_dir("batches", batch_id)
            pdfs = [save_upload(upload, folder) for upload in files]
        elif use_synthetic:
            pdfs = [p for p in (SYNTHETIC_SHEETS / f"{sid}.pdf" for sid in chosen) if p.exists()]
        else:
            raise BadUploadError(
                "No answer sheets were given.", fix="Upload PDFs, or set use_synthetic=true or from_transcripts=true."
            )
        if not pdfs:
            raise DataNotFoundError(f"No answer-sheet PDFs found in {SYNTHETIC_SHEETS}.")
        if ws.demo:
            # 💡 Demo mode can't read handwriting: known synthetic students get their transcript,
            #    anyone else gets blank answers that pause for a human transcription review.
            items = [make_demo_sheet(paper, pdf) for pdf in pdfs]
            how = f"{len(items)} PDF(s); demo mode stands in for the vision model"
        else:
            items = list(pdfs)
            how = f"{len(items)} PDF(s) to transcribe with {settings.transcription_model}"
    student_ids = [getattr(i, "student_id", None) or Path(i).stem for i in items]
    if len(set(student_ids)) != len(student_ids):
        raise BadUploadError(f"Each student can appear only once in a batch: {student_ids}")

    orchestrator = batch_orchestrator(ws, paper)
    job = jobs(request).add(BatchJob(batch_id, paper, student_ids, ws, orchestrator, auto_approve=auto_approve))
    mode = "🧪 demo mode (offline fakes)" if ws.demo else f"grading with {settings.grading_model}"
    job.on_event("queued", None, {"message": f"{how} · {mode}"})
    jobs(request).submit(
        job,
        lambda: grade_batch(
            paper.exam_id,
            items,
            thread_id=job.thread_id,
            exam=paper,
            questions=question_ids,
            orchestrator=orchestrator,
            auto_approve=auto_approve,
            on_event=job.on_event,
        ),
    )
    return BatchCreated(
        batch_id=batch_id, status=job.status, students=student_ids, events_url=f"/batches/{batch_id}/events"
    )


@router.get("/batches", response_model=list[BatchStatus], tags=["batches"])
def get_batches(
    request: Request,
    status: Annotated[str | None, Query(description="Only batches in this state, e.g. paused")] = None,
) -> list[BatchStatus]:
    return [batch_status(job) for job in jobs(request).all() if status is None or job.status == status]


@router.get("/batches/{batch_id}", response_model=BatchStatus, tags=["batches"])
def get_batch(request: Request, batch_id: str) -> BatchStatus:
    """Status, per-student progress and, when paused, the review items with full context."""
    return batch_status(get_job(request, batch_id))


@router.get("/batches/{batch_id}/events", tags=["batches"])
async def batch_events(
    request: Request,
    batch_id: str,
    after: Annotated[int, Query(ge=0, description="Skip events up to this ID")] = 0,
    last_event_id: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    """Server-Sent Events: every progress event as `event: progress`, then one `event: end` with the status.

    💡 SSE is just a long HTTP response in a simple text format (`id:`, `event:`, `data:` lines, a blank
       line between events). Browsers reconnect by themselves and send `Last-Event-ID`, so nothing is missed.
       The stream ends whenever the batch stops (paused, done, failed): reconnect after a review.
    """
    job = get_job(request, batch_id)
    start = int(last_event_id) if (last_event_id or "").isdigit() else after

    async def stream() -> AsyncIterator[str]:
        last, idle = start, 0
        yield "retry: 2000\n\n"
        while True:
            new = job.events_after(last)
            for event in new:
                last = event["id"]
                yield f"id: {last}\nevent: progress\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            if not new:
                if job.status in TERMINAL and not job.events_after(last):
                    end = {"batch_id": job.batch_id, "status": job.status, "last_event_id": last}
                    yield f"event: end\ndata: {json.dumps(end)}\n\n"
                    return
                if await request.is_disconnected():
                    return
                idle += 1
                if idle % 60 == 0:
                    yield ": keep-alive\n\n"  # 💡 a comment line: stops proxies closing an idle stream
                await asyncio.sleep(0.25)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}  # ⚠️ nginx would buffer SSE otherwise
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


@router.post("/batches/{batch_id}/review", response_model=BatchStatus, tags=["batches"])
def review_batch(request: Request, batch_id: str, decision: ReviewDecision) -> BatchStatus:
    """Answer the pending review (mark overrides, or transcription corrections) and resume in the background."""
    from gradewise.orchestrator import resume_batch

    job = get_job(request, batch_id)
    pending = job.pending
    if job.status != "paused" or not pending:
        raise NotWaitingError(f"Batch {batch_id!r} is {job.status}, not waiting for a review.")
    if decision.student_id and decision.student_id != pending["student_id"]:
        raise NotWaitingError(
            f"The pending review is for {pending['student_id']}, not {decision.student_id}.",
            fix="Reload the review queue: someone may have reviewed this student already.",
        )
    if pending["type"] == "human_review":
        resume: dict[str, Any] = {"overrides": decision.overrides, "approved": decision.approved}
    else:
        resume = {"corrections": decision.corrections, "approved": decision.approved}
    jobs(request).submit(
        job,
        lambda: resume_batch(
            job.thread_id, resume, orchestrator=job.orchestrator, on_event=job.on_event, auto_approve=job.auto_approve
        ),
    )
    return batch_status(job)


# 📊 Reports ────────────────────────────────────────────────────────────────────
@router.get("/reports/{exam_id}", response_model=ClassReport, tags=["reports"])
def class_report(request: Request, exam_id: str, demo: DemoQuery = None) -> ClassReport:
    """The class overview: every student's total, grade bands, and per-question averages (hardest first)."""
    ws = workspace(request, demo)
    reports = list_reports(safe_name(exam_id, "exam ID"), db=ws.db)
    if not reports:
        return ClassReport(exam_id=exam_id, count=0)
    pct = [r.percentage for r in reports]
    bands: dict[str, int] = {}
    for r in reports:
        bands[r.grade_band] = bands.get(r.grade_band, 0) + 1
    by_question: dict[str, list] = {}
    for r in reports:
        for g in r.results:
            by_question.setdefault(g.question_id, []).append(g)
    questions = [
        QuestionStat(
            question_id=qid,
            max_marks=results[0].max_marks,
            mean=round(statistics.mean(g.awarded_marks for g in results), 2),
            mean_percent=round(100 * statistics.mean(g.awarded_marks / g.max_marks for g in results), 1),
            flagged=sum(g.needs_review for g in results),
        )
        for qid, results in by_question.items()
    ]
    return ClassReport(
        exam_id=exam_id,
        count=len(reports),
        mean_percent=round(statistics.mean(pct), 1),
        median_percent=round(statistics.median(pct), 1),
        bands=bands,
        flagged=sum(q.flagged for q in questions),
        students=[summarise(r) for r in reports],
        questions=sorted(questions, key=lambda q: q.mean_percent),
    )


@router.get("/reports/{exam_id}/{student_id}", tags=["reports"], response_model=ReportCard)
def student_report(
    request: Request,
    exam_id: str,
    student_id: str,
    format: Annotated[Literal["json", "html", "pdf"], Query()] = "json",  # noqa: A002
    demo: DemoQuery = None,
):
    """One student's report card: JSON (default), the HTML report card, or a PDF download."""
    from gradewise.reports import render_html, render_pdf

    ws = workspace(request, demo)
    report = get_report(safe_name(exam_id, "exam ID"), safe_name(student_id, "student ID"), db=ws.db)
    if format == "json":
        return report
    try:
        exam, _ = resolve_paper(exam_id, ws)
    except GradeWiseError:
        exam = None  # 💡 the card still renders, just without question texts
    if format == "html":
        return HTMLResponse(render_html(report, exam))
    folder = ws.reports_dir(exam_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = render_pdf(report, exam, folder / f"{student_id}.pdf")
    return FileResponse(path, media_type="application/pdf", filename=f"{exam_id}_{student_id}.pdf")


# 💬 Chat ───────────────────────────────────────────────────────────────────────
@router.post("/ask", response_model=AskResponse, tags=["chat"])
def ask_results(request: Request, body: AskRequest) -> AskResponse:
    """Ask the week 5 'chat with results' agent a question about one exam's report cards."""
    from gradewise.reports.qa_agent import NoReportsError, ask

    ws = workspace(request, body.demo)
    safe_name(body.exam_id, "exam ID")
    try:
        if ws.demo:
            answer = ask(
                body.question,
                body.exam_id,
                DemoResultsModel(),
                db=ws.db,
                embeddings=demo_embeddings(),
                persist_dir=ws.vector_dir,
            )
        else:
            answer = ask(body.question, body.exam_id, db=ws.db, persist_dir=ws.vector_dir)
    except NoReportsError as err:
        # 💡 The week 5 error suggests a CLI command; an API caller needs the API's own next step.
        err.fix = "Grade a batch for this exam first (POST /batches, then approve any review), then ask again."
        err.help_ref = "week6/project/README.md"
        raise
    return AskResponse(answer=answer, exam_id=body.exam_id, demo=ws.demo)


# ── 🏗️ The app ──────────────────────────────────────────────────────────────────
def create_app(output_dir: str | Path | None = None, *, demo: bool | None = None) -> FastAPI:
    """Build the API. Tests pass a temporary `output_dir` and demo=True; production uses the defaults.

    Args:
        output_dir: where everything is written (default output/). Demo mode uses <output_dir>/demo/.
        demo: the default mode for requests that don't say (default: the GRADEWISE_DEMO setting).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.jobs.shutdown()  # 💡 on Ctrl+C: stop taking work (running batches are checkpointed)

    app = FastAPI(
        title="GradeWise API",
        version=__version__,
        summary="Grade handwritten CS exams with agentic RAG, with a teacher in the loop.",
        description="The week 6 backend of the course *Agentic RAG with LangChain & LangGraph*. "
        "Every route accepts `demo=true` to run on offline fakes, with no API key.",
        lifespan=lifespan,
        responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    )
    app.state.workspaces = make_workspaces(Path(output_dir) if output_dir else None)
    app.state.default_demo = demo_from_env() if demo is None else demo
    app.state.jobs = JobRegistry()
    app.state.exam_graphs = {}

    @app.exception_handler(GradeWiseError)
    async def gradewise_error(_request: Request, err: GradeWiseError) -> JSONResponse:
        # 💡 The same what / cause / fix / help you see in the terminal, as JSON the UI can show.
        return JSONResponse(error_payload(err), status_code=status_for(err))

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, err: Exception) -> JSONResponse:
        # ⚠️ Never leak a raw traceback to a browser; the full one is in the API's terminal.
        return JSONResponse(error_payload(err), status_code=500)

    app.include_router(router)
    return app


app = create_app()
