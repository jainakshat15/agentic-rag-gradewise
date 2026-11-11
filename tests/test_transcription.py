"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧪 tests/test_transcription.py: the week 4 transcription pipeline   │
│                                                                      │
│  Run:  uv run python -m pytest tests/test_transcription.py -q        │
│  💡 A fake vision model returns canned PageTranscriptions, so these  │
│     tests are free and never touch the network.                      │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from PIL import Image, ImageDraw, ImageFont

from gradewise.config import DATA_DIR
from gradewise.schemas.answers import AnswerSheet
from gradewise.schemas.exam import ExamPaper
from gradewise.transcription import (
    AnswerSegment,
    PageTranscription,
    build_transcription_graph,
    cer,
    combine_confidence,
    double_pass_verify,
    illegible_ratio,
    map_answers,
    needs_review,
    normalise_label,
    preprocess,
    render_page,
    render_pages,
    to_data_url,
    transcribe_page,
    transcribe_sheet,
    wer,
)
from gradewise.transcription.pdf_to_images import estimate_image_tokens, estimate_skew
from gradewise.transcription.vision import build_messages

SYNTHETIC = DATA_DIR / "answer-sheets" / "synthetic"
EXPECTED_EXAM = DATA_DIR / "exams" / "cs_midterm" / "expected_exam.json"
FONT = Path("C:/Windows/Fonts/Inkfree.ttf")


# ── 🧰 Helpers ─────────────────────────────────────────────────────────────
def _font(size: int = 40):
    try:
        return ImageFont.truetype(str(FONT), size)
    except OSError:
        return ImageFont.load_default()


def make_handwritten_pdf(path: Path, pages: list[list[str]]) -> Path:
    """A tiny 'handwritten-looking' PDF: one A5-ish page per list of lines."""
    images = []
    for lines in pages:
        img = Image.new("RGB", (827, 1169), "white")
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(lines):
            draw.text((60, 80 + i * 70), line, fill=(20, 30, 90), font=_font())
        images.append(img)
    images[0].save(path, "PDF", resolution=100, save_all=True, append_images=images[1:])
    return path


def _q(qid: str, marks: float = 2) -> dict:
    return {
        "id": qid,
        "number": qid[1:],
        "text": f"Question {qid}",
        "max_marks": marks,
        "question_type": "short",
        "criteria": [{"id": f"{qid}.c1", "description": "point", "marks": marks}],
        "methodology": "Check the point.",
    }


@pytest.fixture
def exam() -> ExamPaper:
    ids = ["Q1", "Q2", "Q3a", "Q3b", "Q4"]
    return ExamPaper.model_validate(
        {
            "exam_id": "test_exam",
            "title": "Test",
            "subject": "CS201",
            "total_marks": 2 * len(ids),
            "questions": [_q(i) for i in ids],
        }
    )


def seg(label, text, conf=0.95, continued=False, crossed=None, illegible=None) -> AnswerSegment:
    return AnswerSegment(
        label=label,
        text=text,
        confidence=conf,
        continued=continued,
        crossed_out=crossed or [],
        illegible_spans=illegible or [],
    )


def page(n: int, *segments: AnswerSegment, conf: float = 0.95, other=None) -> PageTranscription:
    return PageTranscription(page_number=n, segments=list(segments), other_text=other or [], confidence=conf)


class FakeVisionLLM:
    """Stands in for a vision chat model. Its structured output returns canned PageTranscriptions.

    The page number is read from the prompt (branches run in parallel, so call order is random).
    A second-pass prompt gets `second[page]` if given, else the first-pass page again.
    """

    def __init__(self, first: dict[int, PageTranscription], second: dict[int, PageTranscription] | None = None):
        self.first, self.second = first, second or {}
        self.calls: list[tuple[int, bool]] = []
        self.messages: list[list] = []

    def with_structured_output(self, schema, **kwargs):
        assert schema is PageTranscription
        return RunnableLambda(self._respond)

    def _respond(self, messages):
        self.messages.append(messages)
        text = messages[-1].content[0]["text"]
        number = int(re.search(r"page (\d+)", text)[1])
        second = "second, independent reading" in text
        self.calls.append((number, second))
        parsed = self.second.get(number, self.first[number]) if second else self.first[number]
        raw = AIMessage(content="", usage_metadata={"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200})
        return {"raw": raw, "parsed": parsed, "parsing_error": None}


# ── 🖼️ Rendering ───────────────────────────────────────────────────────────
def test_render_pages_on_a_real_pdf(tmp_path):
    pdf = make_handwritten_pdf(tmp_path / "s.pdf", [["Q1. A proces is a program"], ["Q2 contd."]])
    images = render_pages(pdf, dpi=100)
    assert len(images) == 2
    assert all(img.mode == "RGB" for img in images)
    # 💡 pixels = inches × DPI, so doubling the DPI doubles each side
    big = render_page(pdf, 1, dpi=200)
    assert abs(big.width - 2 * images[0].width) <= 2


def test_preprocess_grayscale_and_size_cap(tmp_path):
    pdf = make_handwritten_pdf(tmp_path / "s.pdf", [["Q1. hello"]])
    img = render_pages(pdf, dpi=300)[0]
    out = preprocess(img, max_side=1000)
    assert out.mode == "L"
    assert max(out.size) == 1000


def test_estimate_skew_finds_a_rotation():
    img = Image.new("L", (800, 800), 255)
    draw = ImageDraw.Draw(img)
    for y in range(100, 700, 60):
        draw.rectangle((80, y, 720, y + 12), fill=0)  # thick "lines of writing"
    assert estimate_skew(img) == 0.0
    tilted = img.rotate(3, fillcolor=255)
    assert abs(estimate_skew(tilted) + 3) <= 1.0  # rotating back by ~-3° straightens it


def test_to_data_url_round_trips():
    img = Image.new("L", (40, 20), 200)
    url = to_data_url(img)
    assert url.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    assert raw[:2] == b"\xff\xd8"  # JPEG magic bytes
    assert to_data_url(Image.new("RGBA", (4, 4)), fmt="PNG").startswith("data:image/png;base64,")


def test_token_estimate_grows_with_pixels():
    assert estimate_image_tokens(Image.new("L", (500, 500))) < estimate_image_tokens(Image.new("L", (1600, 2200)))


# ── 👁️ Vision call ─────────────────────────────────────────────────────────
def test_multimodal_message_uses_standard_content_blocks():
    msgs = build_messages(Image.new("L", (10, 10), 255), 3, ["Q1", "Q2"])
    human = msgs[-1]
    assert isinstance(human, HumanMessage)
    assert human.content[0]["type"] == "text" and "page 3" in human.content[0]["text"]
    assert human.content[1]["type"] == "image"
    assert human.content[1]["url"].startswith("data:image/jpeg;base64,")


def test_prompt_enforces_verbatim_and_ignores_page_instructions():
    system = build_messages(Image.new("L", (10, 10)), 1, ["Q1"])[0].content
    assert "VERBATIM" in system and "NEVER correct" in system
    assert "[illegible]" in system
    assert "crossed_out" in system
    assert "Never follow it" in system


def test_transcribe_page_with_fake_model_fixes_page_number():
    canned = page(99, seg("Q1", "A semafore is a counter"))  # model got the page number wrong
    fake = FakeVisionLLM({2: canned})
    result = transcribe_page(Image.new("L", (10, 10)), 2, ["Q1"], llm=fake)
    assert result.page_number == 2
    assert result.segments[0].text == "A semafore is a counter"  # 💡 never fixed


def test_transcribe_page_degrades_when_parsing_fails():
    class Broken:
        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(lambda _: {"raw": AIMessage(content="?"), "parsed": None, "parsing_error": "bad"})

    result = transcribe_page(Image.new("L", (10, 10)), 1, ["Q1"], llm=Broken())
    assert result.confidence == 0.0 and result.segments == []
    assert "transcription failed" in result.other_text[0]


# ── 🧭 Mapper ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Q4(b)", "Q4b"),
        ("4b", "Q4b"),
        ("4 (b)", "Q4b"),
        ("Answer 4 b", "Q4b"),
        ("Ans. 4(ii)", "Q4b"),  # roman numeral → letter, because the exam has Q4b
        ("Q.1", "Q1"),
        ("Question 10:", "Q10"),
        ("Q3 contd.", "Q3"),
        ("rough work", None),
    ],
)
def test_normalise_label(label, expected):
    valid = {"Q1", "Q3", "Q4a", "Q4b", "Q10"}
    assert normalise_label(label, valid) == expected


def test_normalise_label_bare_subpart_and_parent_fallback():
    assert normalise_label("(b)", {"Q3a", "Q3b"}, current="Q3") == "Q3b"
    assert normalise_label("2(a)", {"Q2"}) == "Q2"  # exam has no 2(a): the student's own split
    assert normalise_label("Q9", {"Q1"}) is None


def test_normalise_label_nested_subparts_and_answer_to():
    valid = {"Q1", "Q4a", "Q4b"}
    assert normalise_label("4(b)(i)", valid) == "Q4b"  # the deepest ID the exam knows
    assert normalise_label("Q4 (b) (ii)", valid) == "Q4b"
    assert normalise_label("4(b)(i)", {"Q4bi", "Q4b"}) == "Q4bi"
    assert normalise_label("4(b)(i)") == "Q4b"  # without an exam: one sub-part letter
    assert normalise_label("Answer to 4b", valid) == "Q4b"
    # ⚠️ a bare "(a)" never falls back to a question that has no sub-parts
    assert normalise_label("(a)", valid, current="Q1") is None


def test_mapper_bare_subpart_after_question_without_subparts_is_unmapped(exam):
    pages = [page(1, seg("Q1", "A process is a program"), seg("Q1 contd.", "in execution"), seg("(a)", "x = 5"))]
    sheet = map_answers(pages, exam, "sam")
    assert sheet.answer_for("Q1").text == "A process is a program\nin execution"  # "(a)" not glued on
    assert sheet.unmapped_text == ["[page 1, label '(a)'] x = 5"]


def test_mapper_continuation_across_pages(exam):
    pages = [
        page(1, seg("Q1", "A process is a program in execution")),
        page(2, seg(None, "and it has its own address space", continued=True), seg("Q2", "Paging")),
    ]
    sheet = map_answers(pages, exam, "sam")
    q1 = sheet.answer_for("Q1")
    assert q1.text == "A process is a program in execution\nand it has its own address space"
    assert q1.pages == [1, 2]
    assert sheet.answer_for("Q2").text == "Paging"


def test_mapper_contd_label_and_out_of_order(exam):
    pages = [
        page(1, seg("4", "Deadlock needs four conditions"), seg("Q1", "A process is")),
        page(2, seg("Q4 contd.", "mutual exclusion, hold and wait"), seg("2.", "Paging splits memory")),
    ]
    sheet = map_answers(pages, exam, "sam")
    assert [a.question_id for a in sheet.answers] == ["Q1", "Q2", "Q3a", "Q3b", "Q4"]  # exam order
    assert sheet.answer_for("Q4").text == "Deadlock needs four conditions\nmutual exclusion, hold and wait"
    assert sheet.answer_for("Q4").pages == [1, 2]
    assert sheet.answer_for("Q2").pages == [2]


def test_mapper_reports_missing_answers_as_blank(exam):
    sheet = map_answers([page(1, seg("Q1", "only this one"))], exam, "sam")
    for qid in ["Q2", "Q3a", "Q3b", "Q4"]:
        answer = sheet.answer_for(qid)
        assert answer.is_blank and answer.pages == []
    assert sheet.unmapped_text == []
    assert sheet.answer_for("Q2").confidence == 1.0  # sure it's blank: nothing went unmapped


def test_mapper_subparts_and_normalised_labels(exam):
    pages = [page(1, seg("3(a)", "FIFO"), seg("(b)", "LRU"), seg("Answer 3 b", "and clock"))]
    sheet = map_answers(pages, exam, "sam")
    assert sheet.answer_for("Q3a").text == "FIFO"
    assert sheet.answer_for("Q3b").text == "LRU\nand clock"


def test_mapper_excludes_crossed_out_text(exam):
    pages = [page(1, seg("Q1", "A thread is light", crossed=["A thread is heavy"])), page(2, seg("Q2", "~~old~~ new"))]
    sheet = map_answers(pages, exam, "sam")
    assert sheet.answer_for("Q1").text == "A thread is light"
    assert sheet.answer_for("Q1").crossed_out == ["A thread is heavy"]
    assert "old" not in sheet.answer_for("Q2").text
    assert sheet.answer_for("Q2").crossed_out == ["old"]


def test_mapper_unmapped_text_and_lower_blank_confidence(exam):
    pages = [
        page(1, seg(None, "stray text before any label"), seg("Q9", "not on this exam"), other=["Name: Sam"]),
        page(2, seg(None, "still under Q9", continued=True), seg("Q1", "real answer")),
    ]
    sheet = map_answers(pages, exam, "sam")
    assert len(sheet.unmapped_text) == 4
    assert any("still under Q9" in t for t in sheet.unmapped_text)  # not glued onto another question
    assert sheet.answer_for("Q1").text == "real answer"
    assert sheet.answer_for("Q2").confidence == 0.5  # blank, but maybe hiding in the unmapped text
    assert needs_review(sheet.answer_for("Q2"))


def test_mapper_name_header_alone_does_not_flag_blanks(exam):
    sheet = map_answers([page(1, seg("Q1", "answer"), other=["Name: Sam  Roll 17"])], exam, "sam")
    assert sheet.unmapped_text == ["[page 1, not an answer] Name: Sam  Roll 17"]
    assert sheet.answer_for("Q2").confidence == 1.0


def test_mapper_keeps_spelling_verbatim(exam):
    sheet = map_answers([page(1, seg("Q1", "semafore is a varible  that  dont"))], exam, "sam")
    assert sheet.answer_for("Q1").text == "semafore is a varible  that  dont"


# ── 🎯 Confidence & CER ────────────────────────────────────────────────────
def test_cer_and_wer_maths():
    assert cer("semaphore", "semaphore") == 0.0
    assert cer("semaphore", "semafore") == pytest.approx(2 / 9)  # 1 substitution (p→f) + 1 deletion (h)
    assert cer("abc", "") == 1.0
    assert cer("", "") == 0.0 and cer("", "x") == 1.0
    assert cer("a  b\nc", "a b c") == 0.0  # whitespace is collapsed
    assert cer("RAM", "ram") > 0 and cer("RAM", "ram", lowercase=True) == 0.0
    assert wer("the cat sat", "the cat sit") == pytest.approx(1 / 3)


def test_illegible_ratio_and_combined_confidence():
    assert illegible_ratio("a b [illegible] d") == 0.25
    assert illegible_ratio("") == 0.0
    assert combine_confidence(0.9) == 0.9
    assert combine_confidence(0.9, illegible=0.25) == 0.45
    assert combine_confidence(0.9, agreement=0.5) == 0.45
    assert combine_confidence(1.0, illegible=0.8) == 0.0


def test_mapper_confidence_penalises_illegible(exam):
    sheet = map_answers([page(1, seg("Q1", "a [illegible] c d", conf=0.9))], exam, "sam")
    assert sheet.answer_for("Q1").confidence == pytest.approx(0.45)
    assert needs_review(sheet.answer_for("Q1"))


def _sheet_and_pages(exam, conf: float):
    first = {1: page(1, seg("Q1", "A deadlock needs mutual exclusion", conf=conf)), 2: page(2, seg("Q2", "Paging"))}
    sheet = map_answers(list(first.values()), exam, "sam")
    images = [Image.new("L", (20, 20), 255)] * 2
    return first, sheet, images


def test_double_pass_agreement_keeps_answer_but_never_raises_confidence(exam):
    first, sheet, images = _sheet_and_pages(exam, conf=0.7)
    fake = FakeVisionLLM(first)  # second pass reads the same text
    out, checks = double_pass_verify(sheet, list(first.values()), exam, llm=fake, images=images)
    assert [c.question_id for c in checks] == ["Q1"]
    assert fake.calls == [(1, True)]  # only the low-confidence page was re-read
    assert checks[0].cer_between_passes == 0.0
    assert out.answer_for("Q1").confidence == pytest.approx(0.7)
    assert out.answer_for("Q2") == sheet.answer_for("Q2")


def test_double_pass_disagreement_lowers_confidence_and_flags(exam):
    first, sheet, images = _sheet_and_pages(exam, conf=0.74)
    second = {1: page(1, seg("Q1", "A dead lok needs mutal exclushun", conf=0.9))}
    fake = FakeVisionLLM(first, second)
    out, checks = double_pass_verify(sheet, list(first.values()), exam, llm=fake, images=images)
    check = checks[0]
    assert check.cer_between_passes > 0.15
    assert check.flagged and check.kept_pass == 2
    assert out.answer_for("Q1").confidence < 0.75
    assert out.answer_for("Q1").confidence < check.second_confidence
    assert out.answer_for("Q1").text == "A dead lok needs mutal exclushun"  # the more confident pass


def test_double_pass_moderate_disagreement_cannot_rescue_a_shaky_answer(exam):
    first, sheet, images = _sheet_and_pages(exam, conf=0.74)
    second = {1: page(1, seg("Q1", "A dead lock need mutal exclution", conf=0.9))}  # ~12% CER
    out, checks = double_pass_verify(sheet, list(first.values()), exam, llm=FakeVisionLLM(first, second), images=images)
    assert checks[0].cer_between_passes < 0.15
    assert needs_review(out.answer_for("Q1"))  # (0.74 + 0.9) / 2 × 0.88 ≈ 0.72


def test_double_pass_skips_confident_sheets(exam):
    first, sheet, _ = _sheet_and_pages(exam, conf=0.95)
    fake = FakeVisionLLM(first)
    out, checks = double_pass_verify(sheet, list(first.values()), exam, llm=fake)  # no images needed
    assert checks == [] and fake.calls == [] and out == sheet


# ── 🧱 The subgraph ────────────────────────────────────────────────────────
def test_subgraph_completes_on_a_two_page_pdf(tmp_path, exam):
    pdf = make_handwritten_pdf(
        tmp_path / "sam.pdf",
        [["Q1. A proces is a program", "in execution"], ["contd. with its own memory", "Q3(a) FIFO"]],
    )
    first = {
        1: page(1, seg("Q1.", "A proces is a program\nin execution")),
        2: page(2, seg("contd.", "with its own memory", continued=True), seg("Q3(a)", "FIFO [illegible]", conf=0.8)),
    }
    second = {2: page(2, seg("contd.", "with its own memory"), seg("Q3(a)", "FIFO queue", conf=0.85))}
    fake = FakeVisionLLM(first, second)

    sheet = transcribe_sheet(pdf, exam, "sam", llm=fake, dpi=100)

    assert isinstance(sheet, AnswerSheet)
    assert sheet.page_count == 2 and sheet.source_file == "sam.pdf"
    assert sheet.answer_for("Q1").text == "A proces is a program\nin execution\nwith its own memory"
    assert sheet.answer_for("Q1").pages == [1, 2]
    assert sheet.answer_for("Q4").is_blank
    assert sorted(c for c in fake.calls if not c[1]) == [(1, False), (2, False)]  # fan-out: every page once
    assert (2, True) in fake.calls  # Q3a was low confidence → page 2 was re-read
    assert needs_review(sheet.answer_for("Q3a"))
    # 💡 the image really went into the message
    assert any(m[-1].content[1]["url"].startswith("data:image/jpeg") for m in fake.messages)


def test_build_graph_has_the_expected_nodes(exam):
    app = build_transcription_graph(FakeVisionLLM({}))
    nodes = set(app.get_graph().nodes)
    assert {"render", "transcribe_page", "map", "verify_low_confidence"} <= nodes


# ── 📦 Course data (once the synthetic sheets exist) ───────────────────────
@pytest.mark.skipif(not EXPECTED_EXAM.exists(), reason="canonical exam not generated yet")
def test_mapper_on_the_canonical_exam():
    exam = ExamPaper.model_validate_json(EXPECTED_EXAM.read_text(encoding="utf-8"))
    ids = [q.id for q in exam.questions]
    sheet = map_answers([page(1, seg(ids[-1], "last first"), seg(ids[0], "then the first"))], exam, "x")
    assert [a.question_id for a in sheet.answers] == ids
    assert sheet.answer_for(ids[0]).text == "then the first"


@pytest.mark.skipif(
    not (SYNTHETIC / "student_01.pdf").exists()
    or not (SYNTHETIC / "transcripts.json").exists()
    or not EXPECTED_EXAM.exists(),
    reason="synthetic answer sheets not generated yet",
)
def test_synthetic_sheet_renders_and_perfect_fake_scores_zero_cer():
    exam = ExamPaper.model_validate_json(EXPECTED_EXAM.read_text(encoding="utf-8"))
    truth = json.loads((SYNTHETIC / "transcripts.json").read_text(encoding="utf-8"))["student_01"]
    images = render_pages(SYNTHETIC / "student_01.pdf", dpi=100)
    assert len(images) >= 1
    # A "perfect" model puts every true answer on page 1: CER against the ground truth must be 0.
    canned = {1: page(1, *[seg(qid, text) for qid, text in truth.items()])}
    canned.update({n: page(n) for n in range(2, len(images) + 1)})
    sheet = transcribe_sheet(SYNTHETIC / "student_01.pdf", exam, "student_01", llm=FakeVisionLLM(canned), dpi=100)
    for qid, text in truth.items():
        assert cer(text, sheet.answer_for(qid).text) == 0.0
