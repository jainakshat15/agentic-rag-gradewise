"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧪 tests/test_exam.py: exam understanding (weeks 1 & 3)             │
│                                                                      │
│  Run:  uv run python -m pytest tests/test_exam.py -q                 │
│  💡 Every LLM here is a fake that returns canned ExamPapers, so the  │
│     tests are free, offline and always give the same answer.         │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from gradewise.config import DATA_DIR
from gradewise.exam.extract import ExamExtractionError, build_messages, extract_exam, extract_with_retry
from gradewise.exam.parse import parse_pdf, read_document, strip_repeated_lines
from gradewise.schemas.exam import ExamPaper

EXAM_DIR = DATA_DIR / "exams" / "cs_midterm"
DOCS = {
    "question_paper": "CS201 Midterm. Q1 (4 marks) Define a process. Q2 (6 marks) Compare TCP and UDP.",
    "marking_scheme": "Q1: program in execution (2), has PCB/state (2). Q2: reliability (3), ordering (3).",
    "methodology": "Ignore spelling. Award partial credit for partially correct points.",
}


def good_paper(exam_id: str = "cs_midterm") -> dict:
    """A tiny, internally consistent exam: 2 questions, 10 marks."""
    return {
        "exam_id": exam_id,
        "title": "CS201 Midterm",
        "subject": "Systems Fundamentals",
        "total_marks": 10,
        "questions": [
            {
                "id": "Q1",
                "number": "1",
                "text": "Define a process.",
                "max_marks": 4,
                "question_type": "definition",
                "criteria": [
                    {"id": "Q1.c1", "description": "A program in execution", "marks": 2},
                    {"id": "Q1.c2", "description": "Has a PCB / state", "marks": 2},
                ],
                "methodology": "Ignore spelling.",
            },
            {
                "id": "Q2",
                "number": "2",
                "text": "Compare TCP and UDP.",
                "max_marks": 6,
                "question_type": "compare",
                "criteria": [
                    {"id": "Q2.c1", "description": "Reliability", "marks": 3},
                    {"id": "Q2.c2", "description": "Ordering", "marks": 3},
                ],
                "methodology": "Partial credit for partially correct points.",
            },
        ],
    }


def bad_paper() -> dict:
    """Valid JSON for the schema, but Q2's criteria only add up to 3 of its 6 marks."""
    paper = good_paper()
    paper["questions"][1]["criteria"] = paper["questions"][1]["criteria"][:1]
    return paper


class FakeStructuredLLM:
    """A stand-in chat model: `with_structured_output(schema)` returns the next canned answer.

    💡 GenericFakeChatModel can't do structured output, so we fake just the piece we use.
       Each canned item is an ExamPaper, a dict (validated like a real model's JSON), or an
       Exception to raise. Every list of messages it receives is recorded in `.calls`.
    """

    def __init__(self, outputs: list) -> None:
        self.outputs = list(outputs)
        self.calls: list[list] = []

    def with_structured_output(self, schema, **kwargs):
        def respond(messages):
            self.calls.append(messages)
            out = self.outputs.pop(0)
            if isinstance(out, Exception):
                raise out
            return out if isinstance(out, schema) else schema.model_validate(out)

        return RunnableLambda(respond)


# ── 📄 parse ─────────────────────────────────────────────────────────────


def test_strip_repeated_lines_removes_headers_and_page_numbers():
    pages = [f"Northbridge Institute\n**Question {n}**\nBody text {n}.\nPage {n} of 3" for n in (1, 2, 3)]
    clean, stripped = strip_repeated_lines(pages)
    assert all("Northbridge" not in p and "Page" not in p for p in clean)
    # 💡 "Question 1/2/3" sits at the top of each page too, but it's content: it must survive
    assert clean[0] == "**Question 1**\nBody text 1."
    assert clean[2] == "**Question 3**\nBody text 3."
    assert stripped == ["northbridge institute", "page # of 3"]


def test_known_footer_is_stripped_even_when_it_appears_once():
    pages = [
        "CS201 Systems Fundamentals · Midterm · Question Paper\n**Question 4**\nBody.\n**Question 5**\n"
        "Northbridge Institute · CS201-MT-2026 Page 1 of 2  (turn over)",
        "CS201 Systems Fundamentals · Midterm · Question Paper\nA system has 32-bit addresses.\n"
        "Northbridge Institute · CS201-MT-2026 Page 2 of 2",
    ]
    clean, stripped = strip_repeated_lines(pages)
    assert clean[0] == "**Question 4**\nBody.\n**Question 5**"  # the footer no longer leaks into Q5
    assert clean[1] == "A system has 32-bit addresses."
    assert "northbridge institute · cs201-mt-2026 page # of 2 (turn over)" in stripped
    # 💡 a one-page PDF still loses its known footer, and the title (with a colon, not a ·) stays
    one, _ = strip_repeated_lines(
        ["## CS201 Systems Fundamentals: Midterm\nQuestion 3\nNorthbridge Institute · School of Computing"]
    )
    assert one == ["## CS201 Systems Fundamentals: Midterm\nQuestion 3"]


@pytest.mark.skipif(not (EXAM_DIR / "question_paper.pdf").exists(), reason="exam pack not generated yet")
def test_question_paper_footer_does_not_leak_into_question_5():
    text = parse_pdf(EXAM_DIR / "question_paper.pdf").text
    assert "Page 1 of 2" not in text and "turn over" not in text
    assert "Question 5**\n\nA system has 32-bit" in text


def test_parse_real_knowledge_base_pdf():
    pdf = DATA_DIR / "knowledge-base" / "unit1_operating_systems.pdf"
    doc = parse_pdf(pdf)
    assert doc.page_count >= 2
    assert "Process Control Block" in doc.text
    assert "Page 1 of" not in doc.text  # the repeated footer is gone
    assert not any(p.text.startswith("CS201 Systems Fundamentals") for p in doc.pages[1:])  # and the header
    assert doc.stripped_lines
    assert "<!-- page 2 -->" in doc.with_page_markers()


def test_read_document_accepts_text_and_paths(tmp_path):
    md = tmp_path / "scheme.md"
    md.write_text("Q1: 2 marks", encoding="utf-8")
    assert read_document("just some text") == "just some text"
    assert read_document(md) == "Q1: 2 marks"
    assert read_document(str(md)) == "Q1: 2 marks"


@pytest.mark.skipif(not (EXAM_DIR / "question_paper.pdf").exists(), reason="exam pack not generated yet")
def test_parse_exam_pack_question_paper():
    doc = parse_pdf(EXAM_DIR / "question_paper.pdf")
    # 💡 question headings sit near page edges: header stripping must never eat them
    for n in range(1, 11):
        assert f"Question {n}" in doc.text


@pytest.mark.skipif(not (EXAM_DIR / "expected_exam.json").exists(), reason="exam pack not generated yet")
def test_graph_end_to_end_on_real_exam_pack(tmp_path):
    from gradewise.db import load_exam
    from gradewise.exam.graph import build_exam_graph

    expected = ExamPaper.model_validate_json((EXAM_DIR / "expected_exam.json").read_text(encoding="utf-8"))
    llm = FakeStructuredLLM([expected])
    graph = build_exam_graph(llm=llm, db=tmp_path / "g.db", write_json=False)
    config = {"configurable": {"thread_id": "real"}}
    result = graph.invoke({"exam_id": "cs_midterm"}, config)  # 💡 parses data/exams/cs_midterm/*.pdf
    assert result["__interrupt__"][0].value["summary"]["total_marks"] == 50
    prompt = llm.calls[0][1].content
    assert "=== MODEL ANSWERS ===" in prompt and "Question 10" in prompt
    graph.invoke(Command(resume={"approved": True}), config)
    assert load_exam("cs_midterm", db=tmp_path / "g.db") == expected


@pytest.mark.skipif(not (EXAM_DIR / "expected_exam.json").exists(), reason="exam pack not generated yet")
def test_expected_exam_is_a_clean_exam_paper():
    paper = ExamPaper.model_validate_json((EXAM_DIR / "expected_exam.json").read_text(encoding="utf-8"))
    assert paper.problems() == []


# ── 🧠 extract ───────────────────────────────────────────────────────────


def test_extract_exam_single_call_and_forces_exam_id():
    llm = FakeStructuredLLM([good_paper(exam_id="something_the_model_made_up")])
    paper = extract_exam(**DOCS, exam_id="cs_midterm", llm=llm)
    assert paper.exam_id == "cs_midterm"
    assert len(llm.calls) == 1
    prompt = llm.calls[0][1].content
    assert "=== MARKING SCHEME ===" in prompt and "Compare TCP and UDP" in prompt


def test_extract_exam_raises_friendly_error_on_schema_failure():
    llm = FakeStructuredLLM([{"exam_id": "x"}])  # missing almost everything
    with pytest.raises(ExamExtractionError):
        extract_exam(**DOCS, llm=llm)


def test_retry_fixes_bad_extraction_on_second_attempt():
    llm = FakeStructuredLLM([bad_paper(), good_paper()])
    paper = extract_with_retry(**DOCS, llm=llm, max_attempts=3)
    assert paper.problems() == []
    assert len(llm.calls) == 2
    # 🔁 the second attempt saw its previous answer AND the exact problem to fix
    feedback = llm.calls[1][-1].content
    assert "Q2: criteria add up to 3.0, but the question is worth 6.0" in feedback
    assert llm.calls[1][-2].type == "ai"


def test_retry_recovers_from_a_schema_error():
    llm = FakeStructuredLLM([{"exam_id": "x", "title": "T"}, good_paper()])
    paper = extract_with_retry(**DOCS, llm=llm)
    assert paper.total_marks == 10
    assert "Schema error" in llm.calls[1][-1].content


def test_retry_gives_up_with_best_paper_after_max_attempts():
    llm = FakeStructuredLLM([bad_paper(), bad_paper()])
    paper = extract_with_retry(**DOCS, llm=llm, max_attempts=2)
    assert paper.problems()  # still flawed: a human must review it
    assert len(llm.calls) == 2


def test_retry_raises_when_nothing_ever_validates():
    llm = FakeStructuredLLM([{"bad": 1}, {"bad": 2}])
    with pytest.raises(ExamExtractionError):
        extract_with_retry(**DOCS, llm=llm, max_attempts=2)


class StrictExamPaper(ExamPaper):
    """A week-1-exercise-style subclass whose own `problems()` adds a rule (every question has topics)."""

    def problems(self) -> list[str]:
        return super().problems() + [f"{q.id}: no reference topics." for q in self.questions if not q.reference_topics]


def test_retry_accepts_an_exampaper_subclass_as_schema():
    first, second = good_paper(), good_paper()
    for q in second["questions"]:
        q["reference_topics"] = ["tcp"]
    llm = FakeStructuredLLM([first, second])
    paper = extract_with_retry(**DOCS, llm=llm, schema=StrictExamPaper)
    assert isinstance(paper, StrictExamPaper) and paper.problems() == []
    assert len(llm.calls) == 2  # the subclass's own problems() drove the retry
    assert "no reference topics" in llm.calls[1][-1].content
    single = extract_exam(**DOCS, llm=FakeStructuredLLM([second]), schema=StrictExamPaper)
    assert isinstance(single, StrictExamPaper)


def test_messages_have_no_feedback_on_first_attempt():
    messages = build_messages("qp", "ms", "meth")
    assert [m.type for m in messages] == ["system", "human"]


# ── 🗄️ db ────────────────────────────────────────────────────────────────


def test_db_round_trip(tmp_path):
    from gradewise.db import ExamNotFoundError, delete_exam, list_exams, load_exam, save_exam

    db = tmp_path / "test.db"
    paper = ExamPaper.model_validate(good_paper())
    save_exam(paper, approved_by="emma", db=db)
    assert load_exam("cs_midterm", db=db) == paper

    # saving again updates in place rather than duplicating
    changed = paper.model_copy(update={"title": "CS201 Midterm (v2)"})
    save_exam(changed, db=db)
    records = list_exams(db=db)
    assert [r.exam_id for r in records] == ["cs_midterm"]
    assert records[0].title == "CS201 Midterm (v2)" and records[0].approved_by == "emma"
    assert records[0].question_count == 2

    assert delete_exam("cs_midterm", db=db)
    with pytest.raises(ExamNotFoundError):
        load_exam("cs_midterm", db=db)


# ── 🕸️ graph ─────────────────────────────────────────────────────────────


def _start(graph, thread: str = "t1"):
    config = {"configurable": {"thread_id": thread}}
    result = graph.invoke({"exam_id": "cs_midterm", "documents": DOCS, "max_attempts": 3}, config)
    return config, result


def test_graph_retries_pauses_for_review_then_stores(tmp_path):
    from gradewise.db import load_exam
    from gradewise.exam.graph import build_exam_graph

    db = tmp_path / "g.db"
    llm = FakeStructuredLLM([bad_paper(), good_paper()])
    graph = build_exam_graph(llm=llm, db=db, write_json=False)

    config, result = _start(graph)
    assert "__interrupt__" in result  # ⏸ paused for Emma
    payload = result["__interrupt__"][0].value
    assert payload["summary"]["question_count"] == 2 and payload["problems"] == []
    assert graph.get_state(config).next == ("admin_review",)
    assert result["attempts"] == 2  # bad → retry → good

    final = graph.invoke(Command(resume={"approved": True}), config)
    assert final["stored"] is True
    assert load_exam("cs_midterm", db=db).total_marks == 10


def test_graph_applies_admin_edits(tmp_path):
    from gradewise.db import load_exam
    from gradewise.exam.graph import build_exam_graph

    db = tmp_path / "g.db"
    graph = build_exam_graph(llm=FakeStructuredLLM([good_paper()]), db=db, write_json=False)
    config, _ = _start(graph)
    graph.invoke(Command(resume={"approved": True, "edits": {"title": "CS201 Midterm, edited by Emma"}}), config)
    assert load_exam("cs_midterm", db=db).title == "CS201 Midterm, edited by Emma"


def test_graph_rejection_stores_nothing(tmp_path):
    from gradewise.db import ExamNotFoundError, load_exam
    from gradewise.exam.graph import build_exam_graph

    db = tmp_path / "g.db"
    graph = build_exam_graph(llm=FakeStructuredLLM([good_paper()]), db=db, write_json=False)
    config, _ = _start(graph)
    final = graph.invoke(Command(resume={"approved": False}), config)
    assert not final.get("stored")
    with pytest.raises(ExamNotFoundError):
        load_exam("cs_midterm", db=db)


def test_graph_sends_flawed_exam_to_review_after_max_attempts(tmp_path):
    from gradewise.exam.graph import build_exam_graph

    graph = build_exam_graph(llm=FakeStructuredLLM([bad_paper()] * 3), db=tmp_path / "g.db", write_json=False)
    _, result = _start(graph)
    assert result["attempts"] == 3
    assert result["__interrupt__"][0].value["problems"]  # Emma sees what's still wrong


def test_graph_survives_restart_with_sqlite_checkpointer(tmp_path):
    from gradewise.db import load_exam
    from gradewise.exam.graph import build_exam_graph, sqlite_checkpointer

    db, ckpt = tmp_path / "g.db", tmp_path / "ckpt.db"
    first = build_exam_graph(sqlite_checkpointer(ckpt), llm=FakeStructuredLLM([good_paper()]), db=db, write_json=False)
    config, _ = _start(first, thread="restart")

    # 💡 a brand-new graph object (think: Python restarted) picks up the paused thread
    second = build_exam_graph(sqlite_checkpointer(ckpt), llm=FakeStructuredLLM([]), db=db, write_json=False)
    second.invoke(Command(resume={"approved": True}), config)
    assert load_exam("cs_midterm", db=db).exam_id == "cs_midterm"


def test_rerunning_a_finished_thread_starts_fresh(tmp_path):
    from gradewise.exam.graph import build_exam_graph, start_input

    llm = FakeStructuredLLM([bad_paper(), bad_paper(), good_paper(), bad_paper(), good_paper()])
    graph = build_exam_graph(llm=llm, db=tmp_path / "g.db", write_json=False)
    config = {"configurable": {"thread_id": "same"}}
    first = graph.invoke(start_input(documents=DOCS), config)
    assert first["attempts"] == 3
    graph.invoke(Command(resume={"approved": False}), config)

    # 💡 same thread, new run: attempts and problems must not leak in from last time
    second = graph.invoke(start_input(documents=DOCS), config)
    assert second["attempts"] == 2 and second["problems"] == []
    assert not second["stored"]


def test_exam_summary_is_json_serialisable():
    from gradewise.exam.summary import exam_summary, print_exam_summary

    paper = ExamPaper.model_validate(bad_paper())
    summary = exam_summary(paper)
    json.dumps(summary)
    assert summary["criteria_count"] == 3 and summary["problems"]
    print_exam_summary(paper)  # must not crash on a flawed exam
