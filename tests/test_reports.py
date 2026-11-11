"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧪 tests/test_reports.py: report cards, storage, chat-with-results  │
│                                                                      │
│  Run:  uv run python -m pytest tests/test_reports.py -q              │
│  💡 Free: a temp SQLite file, a temp Chroma with fake embeddings,    │
│     and scripted fake chat models.                                   │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, ToolMessage

from gradewise.config import DATA_DIR
from gradewise.db import ReportNotFoundError, delete_reports, get_report, list_reports, save_report
from gradewise.grading.fakes import ScriptedChatModel, tool_call_message
from gradewise.ingestion import get_vectorstore
from gradewise.reports import (
    render_html,
    render_pdf,
    report_documents,
    reports_collection,
    rule_based_feedback,
    save_report_files,
    search_reports,
    store_report,
    write_feedback,
)
from gradewise.reports.qa_agent import ask, build_report_agent, question_statistics
from gradewise.schemas import CriterionResult, ExamPaper, GradeResult, ReportCard

EXPECTED_EXAM = DATA_DIR / "exams" / "cs_midterm" / "expected_exam.json"
CITATION = "unit2_computer_networks.pdf p6 §2.4 Subnets"


@pytest.fixture(scope="module")
def exam() -> ExamPaper:
    if not EXPECTED_EXAM.exists():
        pytest.skip("canonical exam not generated yet")
    return ExamPaper.model_validate_json(EXPECTED_EXAM.read_text(encoding="utf-8"))


def make_report(exam: ExamPaper, student_id: str = "student_04", share: float = 0.75, weak: str = "Q7a") -> ReportCard:
    """A report where every question earns `share` of its criteria, except `weak`, which earns 0 and is flagged."""
    results = []
    for q in exam.questions:
        criteria = [
            CriterionResult(
                criterion_id=c.id,
                awarded=0.0 if q.id == weak else c.marks * (1 if i < len(q.criteria) * share else 0),
                justification=f"Quoted: '{c.keywords[0] if c.keywords else 'point'}'",
                evidence=[CITATION] if q.id == weak else [],
            )
            for i, c in enumerate(q.criteria)
        ]
        results.append(
            GradeResult(
                question_id=q.id,
                awarded_marks=sum(c.awarded for c in criteria),
                max_marks=q.max_marks,
                criteria=criteria,
                feedback="Forgot to subtract the network and broadcast addresses." if q.id == weak else "Good.",
                confidence=0.6 if q.id == weak else 0.9,
                needs_review=q.id == weak,
                review_reason="low grader confidence (0.60)" if q.id == weak else None,
            )
        )
    return ReportCard.from_results(student_id, exam.exam_id, results)


# ── 📄 Writing ──────────────────────────────────────────────────────────────
def test_render_html_shows_the_key_fields(exam):
    report = make_report(exam).model_copy(update={"strengths": ["Strong <b>DNS</b> answer"], "improvements": ["Q7a"]})
    page = render_html(report, exam)
    assert page.startswith("<!doctype html>") and "<style>" in page and "http" not in page.split("<body>")[0]
    for text in ("student_04", f"{report.total:g}", report.grade_band, "7(a)", "Q7a.c1", CITATION, "Needs review"):
        assert text in page
    assert "Strong &lt;b&gt;DNS&lt;/b&gt; answer" in page  # ⚠️ student/model text is escaped
    assert "#7C5CFF" in page and "#22D3EE" in page and "#0B1020" in page  # the course palette


def test_html_escapes_hostile_student_text(exam):
    report = make_report(exam)
    first = report.results[0]
    evil = first.model_copy(update={"feedback": "<script>alert('pwned')</script>"})
    page = render_html(report.model_copy(update={"results": [evil, *report.results[1:]]}), exam)
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_render_pdf_writes_a_readable_pdf(exam, tmp_path):
    import pymupdf

    path = render_pdf(make_report(exam), exam, tmp_path / "card.pdf")
    doc = pymupdf.open(path)
    text = "".join(page.get_text() for page in doc)
    assert doc.page_count >= 1 and "student_04" in text and "Question by question" in text and "Q7a.c1" in text
    pix = doc[0].get_pixmap(dpi=40)  # 💡 the "render a page to PNG and look at it" check, automated
    assert pix.width > 100 and pix.height > 100


def test_save_report_files(exam, tmp_path):
    paths = save_report_files(make_report(exam), exam, tmp_path, pdf=False)
    assert paths["html"].exists() and "pdf" not in paths


def test_rule_based_feedback(exam):
    report = rule_based_feedback(make_report(exam), exam)
    assert report.strengths and report.improvements
    assert any("Q7a" in s for s in report.improvements)
    assert not any("Q7a" in s for s in report.strengths)


def test_write_feedback_uses_structured_output(exam):
    llm = ScriptedChatModel(
        messages=iter(
            [tool_call_message("ReportFeedback", {"strengths": ["Great DNS walk-through"], "improvements": ["Q7a"]})]
        )
    )
    report = write_feedback(make_report(exam), exam, llm=llm)
    assert report.strengths == ["Great DNS walk-through"] and report.improvements == ["Q7a"]


def test_write_feedback_falls_back_to_rules(exam):
    llm = ScriptedChatModel(messages=iter([AIMessage("I refuse to use your schema")]))
    report = write_feedback(make_report(exam), exam, llm=llm)
    assert report.improvements  # rule-based fallback filled it in


# ── 💾 Storing ──────────────────────────────────────────────────────────────
def test_store_round_trip(exam, tmp_path):
    db = tmp_path / "g.db"
    report = make_report(exam)
    save_report(report, db=db)
    save_report(report.model_copy(update={"strengths": ["updated"]}), db=db)  # same student → replaced
    save_report(make_report(exam, "student_01", share=1.0, weak="none"), db=db)
    back = get_report(exam.exam_id, "student_04", db=db)
    assert back.strengths == ["updated"] and back.total == report.total and back.results[0] == report.results[0]
    assert [r.student_id for r in list_reports(exam.exam_id, db=db)] == ["student_01", "student_04"]
    with pytest.raises(ReportNotFoundError):
        get_report(exam.exam_id, "nobody", db=db)
    assert delete_reports(exam.exam_id, db=db) == 2 and list_reports(exam.exam_id, db=db) == []


def test_reports_are_indexed_one_chunk_per_question(exam, tmp_path):
    emb = DeterministicFakeEmbedding(size=32)
    vdb = tmp_path / "vdb"
    report = make_report(exam)
    docs = report_documents(report, exam)
    assert len(docs) == len(exam.questions)
    q7a = next(d for d in docs if d.metadata["question_id"] == "Q7a")
    assert q7a.metadata["student_id"] == "student_04" and q7a.metadata["needs_review"] is True
    assert "Flagged for review" in q7a.page_content and "subnet" in q7a.page_content.lower()

    for _ in range(2):  # 💡 storing twice must not duplicate chunks (stable IDs → upsert)
        store_report(report, exam, db=tmp_path / "g.db", embeddings=emb, persist_dir=vdb)
    store_report(make_report(exam, "student_01"), exam, db=tmp_path / "g.db", embeddings=emb, persist_dir=vdb)
    store = get_vectorstore(reports_collection(exam.exam_id), embeddings=emb, persist_dir=vdb)
    assert store._collection.count() == 2 * len(exam.questions)
    hits = search_reports(exam.exam_id, "subnet hosts", student_id="student_04", embeddings=emb, persist_dir=vdb)
    assert hits and {d.metadata["student_id"] for d in hits} == {"student_04"}
    only = search_reports(exam.exam_id, "x", question_id="Q7a", k=10, embeddings=emb, persist_dir=vdb)
    assert {d.metadata["question_id"] for d in only} == {"Q7a"} and len(only) == 2


# ── 💬 Chat with results ────────────────────────────────────────────────────
@pytest.fixture
def stored_class(exam, tmp_path):
    db, vdb, emb = tmp_path / "g.db", tmp_path / "vdb", DeterministicFakeEmbedding(size=32)
    for sid, share in [("student_01", 1.0), ("student_04", 0.75), ("student_07", 0.5)]:
        store_report(make_report(exam, sid, share=share), exam, db=db, embeddings=emb, persist_dir=vdb)
    return {"db": db, "embeddings": emb, "persist_dir": vdb}


def test_question_statistics_puts_the_hardest_first(exam, stored_class):
    stats = question_statistics(list_reports(exam.exam_id, db=stored_class["db"]))
    assert stats.index[0] == "Q7a" and stats.loc["Q7a", "mean"] == 0 and stats.loc["Q7a", "flagged"] == 3
    assert {"mean", "median", "min", "max", "mean_pct", "distribution"} <= set(stats.columns)


def test_report_agent_uses_its_tools(exam, stored_class):
    llm = ScriptedChatModel(
        messages=iter(
            [
                tool_call_message("class_statistics", {}),
                tool_call_message("get_student_report", {"student_id": "student_04"}),
                tool_call_message("search_reports", {"query": "subnet", "question_id": "Q7a"}),
                tool_call_message("list_flagged", {}),
                AIMessage("Q7a was the hardest: student_04 forgot to subtract the network and broadcast addresses."),
            ]
        )
    )
    agent = build_report_agent(exam.exam_id, llm, **stored_class)
    state = agent.invoke({"messages": [{"role": "user", "content": "Why did student_04 lose marks on Q7a?"}]})
    stats, student, search, flagged = (m.content for m in state["messages"] if isinstance(m, ToolMessage))
    rows = [line for line in stats.splitlines() if line.startswith("Q")]
    assert "3 students" in stats and rows[0].startswith("Q7a")  # hardest question first
    assert "Q7a: 0/2" in student and "Forgot to subtract" in student
    assert "Q7a" in search
    assert flagged.count("Q7a") == 3
    assert "hardest" in state["messages"][-1].text


def test_ask_returns_text(exam, stored_class):
    llm = ScriptedChatModel(messages=iter([AIMessage("Q7a.")]))
    assert ask("Which question was hardest?", exam.exam_id, llm, **stored_class) == "Q7a."
