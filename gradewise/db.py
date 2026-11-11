"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 3 · Day 5: GradeWise's database                             │
│                                                                      │
│  What this file does                                                 │
│    One SQLite file, output/gradewise.db, managed with SQLModel       │
│    (Pydantic + SQLAlchemy). Today it stores approved exams; in       │
│    week 5 the report cards join them.                                │
│                                                                      │
│      save_exam(paper)      → insert or update an ExamPaper           │
│      load_exam("cs_midterm") → the ExamPaper back, fully validated   │
│      list_exams()          → every stored exam (newest first)        │
│      save_report / get_report / list_reports  → report cards (wk 5)  │
│                                                                      │
│  ⏮ Built on: gradewise/schemas/exam.py                               │
│  ⏭ Used by: gradewise/exam/graph.py (store node), week 5 reports,    │
│             week 6 API and dashboard                                 │
╰──────────────────────────────────────────────────────────────────────╯

🧱 HOW TO ADD A TABLE (e.g. week 5 report cards)
   1. Define a model below `ExamRecord`, with `table=True` and a foreign key to the exam:

          class ReportRecord(SQLModel, table=True):
              __tablename__ = "reports"
              id: int | None = Field(default=None, primary_key=True)
              exam_id: str = Field(foreign_key="exams.exam_id", index=True)
              student_id: str = Field(index=True)
              report_json: str = Field(sa_column=Column(Text, nullable=False))
              created_at: datetime = Field(default_factory=_now)

   2. Write a `save_report` / `load_report` pair that opens `with session(db) as s:`, exactly like
      `save_exam` / `load_exam`. Store the Pydantic object as JSON (`model_dump_json()`), and copy out
      only the few columns you want to filter or sort on (student_id, total...).
   3. That's it: `init_db()` calls `SQLModel.metadata.create_all`, which creates every table
      defined in this module the next time any function here runs.

   ⚠️ `create_all` creates *missing* tables; it never alters an existing one. If you change the
      columns of a table that already exists, delete output/gradewise.db (it's rebuildable) or,
      🚀 PRO, use Alembic migrations.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path


def _import_sqlmodel():
    """Import SQLModel, falling back to SQLAlchemy's pure-Python modules if compiled ones are blocked.

    ⚠️ Windows gotcha: on locked-down machines (Windows "Application Control" / WDAC), freshly
       installed compiled extensions (.pyd files) can be blocked, and `import sqlalchemy` dies with
       "DLL load failed ... An Application Control policy has blocked this file". SQLAlchemy ships a
       pure-Python twin of each compiled module, so we load those instead. Slightly slower, same results.
    """
    try:
        import sqlmodel
    except ImportError as exc:
        if "DLL load failed" not in str(exc):
            raise
        import importlib.abc
        import importlib.util
        import sys

        class _PurePythonSQLAlchemy(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if not (name.startswith("sqlalchemy.") and name.endswith("_cy")):
                    return None
                for folder in path or []:
                    source = Path(folder) / f"{name.rsplit('.', 1)[1]}.py"
                    if source.exists():
                        return importlib.util.spec_from_file_location(name, source)
                return None

        for module in [m for m in sys.modules if m == "sqlalchemy" or m.startswith(("sqlalchemy.", "sqlmodel"))]:
            del sys.modules[module]  # 💡 forget the half-finished import before trying again
        sys.meta_path.insert(0, _PurePythonSQLAlchemy())
        import sqlmodel
    return sqlmodel


_import_sqlmodel()

from sqlalchemy import Column, Text  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402
from sqlmodel import Field, Session, SQLModel, create_engine, select  # noqa: E402

from gradewise.config import OUTPUT_DIR  # noqa: E402
from gradewise.errors import GradeWiseError  # noqa: E402
from gradewise.schemas.exam import ExamPaper  # noqa: E402

DB_PATH = OUTPUT_DIR / "gradewise.db"

# 💡 Every function below takes `db=`: a file path, a ready-made Engine, or None for
#    output/gradewise.db. Tests pass a temporary path, so they never touch your real data.
DB = Path | str | Engine | None


class ExamNotFoundError(GradeWiseError):
    title = "Exam not found"
    cause = "No exam with this ID has been approved and stored yet."
    fix = "Run the exam workflow first: `uv run week3/project/exam_graph.py`, and approve the exam."
    help_ref = "week3/project/README.md"


def _now() -> datetime:
    return datetime.now(UTC)


class ExamRecord(SQLModel, table=True):
    """One approved exam. The full ExamPaper lives in `paper_json`; the other columns are for listing."""

    __tablename__ = "exams"

    exam_id: str = Field(primary_key=True)
    title: str
    subject: str
    total_marks: float
    question_count: int
    approved_by: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    # 💡 Store the whole Pydantic object as JSON: the schema can grow without a migration.
    paper_json: str = Field(sa_column=Column(Text, nullable=False))

    def to_paper(self) -> ExamPaper:
        return ExamPaper.model_validate_json(self.paper_json)


@lru_cache(maxsize=8)
def _engine_for(path: str) -> Engine:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{Path(path).as_posix()}")
    SQLModel.metadata.create_all(engine)  # 💡 creates every table defined in this module
    return engine


def get_engine(db: DB = None) -> Engine:
    """The SQLAlchemy engine for `db` (default output/gradewise.db). Tables are created on first use."""
    if isinstance(db, Engine):
        SQLModel.metadata.create_all(db)
        return db
    return _engine_for(str(Path(db) if db else DB_PATH))


def init_db(db: DB = None) -> Engine:
    """Create the database file and all tables. Safe to call any number of times."""
    return get_engine(db)


@contextmanager
def session(db: DB = None) -> Iterator[Session]:
    """`with session() as s:` : a database session that is always closed afterwards."""
    with Session(get_engine(db)) as s:
        yield s


# ── 📝 Exams ─────────────────────────────────────────────────────────────


def save_exam(paper: ExamPaper, *, approved_by: str | None = None, db: DB = None) -> ExamRecord:
    """Insert the exam, or replace it if one with the same exam_id is already stored."""
    with session(db) as s:
        record = s.get(ExamRecord, paper.exam_id)
        if record is None:
            record = ExamRecord(
                exam_id=paper.exam_id, title="", subject="", total_marks=0, question_count=0, paper_json=""
            )
        record.title = paper.title
        record.subject = paper.subject
        record.total_marks = paper.total_marks
        record.question_count = len(paper.questions)
        record.approved_by = approved_by or record.approved_by
        record.updated_at = _now()
        record.paper_json = paper.model_dump_json()
        s.add(record)
        s.commit()
        s.refresh(record)
        return record


def load_exam(exam_id: str, *, db: DB = None) -> ExamPaper:
    """The stored ExamPaper for `exam_id`. Raises ExamNotFoundError if it isn't there."""
    with session(db) as s:
        record = s.get(ExamRecord, exam_id)
        if record is None:
            raise ExamNotFoundError(f'There is no exam "{exam_id}" in the database.')
        return record.to_paper()


def list_exams(*, db: DB = None) -> list[ExamRecord]:
    """Every stored exam, most recently updated first."""
    with session(db) as s:
        return list(s.exec(select(ExamRecord).order_by(ExamRecord.updated_at.desc())).all())


def delete_exam(exam_id: str, *, db: DB = None) -> bool:
    """Remove an exam. Returns False if it wasn't there."""
    with session(db) as s:
        record = s.get(ExamRecord, exam_id)
        if record is None:
            return False
        s.delete(record)
        s.commit()
        return True


# ── 📊 Report cards (week 5) ─────────────────────────────────────────────
# 💡 Added in week 5 exactly as the recipe at the top of this file describes. One row per
#    (exam, student); re-grading a student replaces their row. We skip a hard foreign key to
#    `exams` so you can grade against an exam JSON that was never stored (handy in notebooks).


class ReportNotFoundError(GradeWiseError):
    title = "Report not found"
    cause = "This student hasn't been graded for this exam yet (or the ID is misspelled)."
    fix = "Grade them first: `uv run week5/project/orchestrator.py --from-transcripts`."
    help_ref = "week5/project/README.md"


class ReportRecord(SQLModel, table=True):
    """One student's report card. The full ReportCard lives in `report_json`; the rest is for listing."""

    __tablename__ = "reports"

    report_id: str = Field(primary_key=True)  # "<exam_id>:<student_id>"
    exam_id: str = Field(index=True)
    student_id: str = Field(index=True)
    total: float
    max_total: float
    percentage: float
    grade_band: str
    needs_review: bool = False
    reviewed_by: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    report_json: str = Field(sa_column=Column(Text, nullable=False))

    def to_report(self):
        from gradewise.schemas.report import ReportCard

        return ReportCard.model_validate_json(self.report_json)


def save_report(report, *, reviewed_by: str | None = None, db: DB = None) -> ReportRecord:
    """Insert a ReportCard, or replace the stored one for the same exam and student."""
    report_id = f"{report.exam_id}:{report.student_id}"
    with session(db) as s:
        record = s.get(ReportRecord, report_id)
        if record is None:
            record = ReportRecord(
                report_id=report_id,
                exam_id=report.exam_id,
                student_id=report.student_id,
                total=0,
                max_total=0,
                percentage=0,
                grade_band="",
                report_json="",
            )
        record.total = report.total
        record.max_total = report.max_total
        record.percentage = report.percentage
        record.grade_band = report.grade_band
        record.needs_review = report.needs_review
        record.reviewed_by = reviewed_by or record.reviewed_by
        record.updated_at = _now()
        record.report_json = report.model_dump_json()
        s.add(record)
        s.commit()
        s.refresh(record)
        return record


def get_report(exam_id: str, student_id: str, *, db: DB = None):
    """The stored ReportCard for one student. Raises ReportNotFoundError if there isn't one."""
    with session(db) as s:
        record = s.get(ReportRecord, f"{exam_id}:{student_id}")
        if record is None:
            raise ReportNotFoundError(f'There is no report for "{student_id}" on exam "{exam_id}".')
        return record.to_report()


def list_reports(exam_id: str, *, db: DB = None) -> list:
    """Every stored ReportCard for an exam, sorted by student ID."""
    with session(db) as s:
        rows = s.exec(select(ReportRecord).where(ReportRecord.exam_id == exam_id).order_by(ReportRecord.student_id))
        return [r.to_report() for r in rows.all()]


def delete_reports(exam_id: str, *, db: DB = None) -> int:
    """Remove every report for an exam (e.g. before a clean re-run). Returns how many were deleted."""
    with session(db) as s:
        rows = s.exec(select(ReportRecord).where(ReportRecord.exam_id == exam_id)).all()
        for row in rows:
            s.delete(row)
        s.commit()
        return len(rows)
