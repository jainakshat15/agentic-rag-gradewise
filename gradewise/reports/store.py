"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 5 · Day 3: Storing reports, twice                           │
│                                                                      │
│  What this file does                                                 │
│    Every report card is saved in TWO places, for two kinds of        │
│    question Asha will ask:                                           │
│                                                                      │
│      SQLite (gradewise/db.py)  → exact questions: "what did          │
│                                  student_04 get?", averages, lists   │
│      Chroma reports_<exam_id>  → fuzzy questions: "who struggled     │
│                                  with subnetting?" One chunk per     │
│                                  student-question, with metadata     │
│                                                                      │
│      store_report(report, exam)  → both at once                      │
│      search_reports(exam_id, q)  → semantic search over the chunks   │
│                                                                      │
│  🔁 RECAP: the same Chroma + metadata ideas as the week 2 knowledge  │
│     base, now pointed at GradeWise's OWN output.                     │
│  ⏭ Used by: orchestrator.py (store node), qa_agent.py, week 6 API   │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from gradewise.db import DB, get_report, list_reports, save_report
from gradewise.schemas.exam import ExamPaper
from gradewise.schemas.report import ReportCard

__all__ = [
    "get_report",
    "index_report",
    "list_reports",
    "report_documents",
    "reports_collection",
    "save_report",
    "search_reports",
    "store_report",
]


def reports_collection(exam_id: str) -> str:
    """The Chroma collection that holds one exam's reports: "reports_cs_midterm"."""
    return f"reports_{exam_id}"


def report_documents(report: ReportCard, exam: ExamPaper | None = None) -> list[Document]:
    """One Document per (student, question), written so that a search for a TOPIC finds it.

    💡 The text mixes the question, the marks, the examiner's justifications and the feedback.
       The metadata (student_id, question_id, marks, needs_review) lets the Q&A agent filter exactly.
    """
    questions = {q.id: q for q in exam.questions} if exam else {}
    docs = []
    for r in report.results:
        q = questions.get(r.question_id)
        lines = [
            f"Student {report.student_id} · Question {r.question_id}"
            + (f" ({q.number})" if q else "")
            + f" · {r.awarded_marks:g}/{r.max_marks:g} marks",
        ]
        if q:
            lines.append(f"Question: {q.text}")
            if q.reference_topics:
                lines.append(f"Topics: {', '.join(q.reference_topics)}")
        lines.append(f"Feedback: {r.feedback}")
        for c in r.criteria:
            lines.append(f"- {c.criterion_id}: {c.awarded:g} marks. {c.justification}")
        if r.needs_review:
            lines.append(f"Flagged for review: {r.review_reason}")
        docs.append(
            Document(
                page_content="\n".join(lines),
                id=f"{report.exam_id}:{report.student_id}:{r.question_id}",
                metadata={
                    "exam_id": report.exam_id,
                    "student_id": report.student_id,
                    "question_id": r.question_id,
                    "awarded": r.awarded_marks,
                    "max_marks": r.max_marks,
                    "percentage": round(100 * r.awarded_marks / r.max_marks, 1),
                    "needs_review": r.needs_review,
                    "grade_band": report.grade_band,
                },
            )
        )
    return docs


def _store(exam_id: str, embeddings: Any = None, persist_dir: Path | str | None = None):
    from gradewise.ingestion.store import get_vectorstore

    return get_vectorstore(reports_collection(exam_id), embeddings=embeddings, persist_dir=persist_dir)


def index_report(
    report: ReportCard,
    exam: ExamPaper | None = None,
    *,
    embeddings: Any = None,
    persist_dir: Path | str | None = None,
) -> int:
    """Add (or replace) one report's chunks in Chroma. Returns how many chunks were written.

    💡 The chunk IDs are stable ("cs_midterm:student_04:Q7a"), and Chroma UPSERTS by ID, so
       re-grading a student overwrites their old chunks instead of duplicating them.
    """
    docs = report_documents(report, exam)
    if docs:
        _store(report.exam_id, embeddings, persist_dir).add_documents(docs, ids=[d.id for d in docs])
    return len(docs)


def store_report(
    report: ReportCard,
    exam: ExamPaper | None = None,
    *,
    db: DB = None,
    index: bool = True,
    reviewed_by: str | None = None,
    embeddings: Any = None,
    persist_dir: Path | str | None = None,
) -> ReportCard:
    """Save a report to SQLite and (unless index=False) to the Chroma reports collection."""
    save_report(report, reviewed_by=reviewed_by, db=db)
    if index:
        index_report(report, exam, embeddings=embeddings, persist_dir=persist_dir)
    return report


def search_reports(
    exam_id: str,
    query: str,
    *,
    k: int = 5,
    student_id: str | None = None,
    question_id: str | None = None,
    embeddings: Any = None,
    persist_dir: Path | str | None = None,
) -> list[Document]:
    """Semantic search over one exam's report chunks, optionally filtered to a student or a question."""
    from gradewise.rag.retrievers import to_chroma_where

    where = {key: value for key, value in (("student_id", student_id), ("question_id", question_id)) if value}
    store = _store(exam_id, embeddings, persist_dir)
    if store._collection.count() == 0:
        return []
    return store.similarity_search(query, k=k, filter=to_chroma_where(where))
