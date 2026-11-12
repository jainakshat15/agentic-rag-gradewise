"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📝 scripts/build_exam_pack.py: the CS201 midterm → PDFs             │
│                                                                      │
│  Turns the Markdown in scripts/exam_pack_source/ into the three      │
│  documents Emma hands GradeWise:                                     │
│    question_paper.pdf · marking_scheme.pdf · methodology.pdf         │
│  in data/exams/cs_midterm/, then checks them against the canonical   │
│  answer key, expected_exam.json (an ExamPaper).                      │
│                                                                      │
│  Run:  uv run scripts/build_exam_pack.py                             │
│  ⏮ Built on: scripts/build_knowledge_base.py (same PDF engine)      │
│  ⏭ Used by: week1/day5 + week3/day5 (exam extraction), week5        │
│  💡 Learners never need to run this: the PDFs are committed.         │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf
from markdown_it import MarkdownIt

from gradewise.console import banner, step, success, warn
from gradewise.errors import DataNotFoundError
from gradewise.schemas import ExamPaper

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(__file__).resolve().parent / "exam_pack_source"
OUTPUT = ROOT / "data" / "exams" / "cs_midterm"
EXPECTED = OUTPUT / "expected_exam.json"
MODEL_ANSWERS = OUTPUT / "model_answers.md"

PAGE = pymupdf.paper_rect("a4")
MARGIN = (56, 60, -56, -64)  # left, top, right, bottom (points)

# 💡 A plainer, more "official" look than the course notes: exam papers are black and white.
# ⚠️ PyMuPDF's Story engine draws CSS background colours on table cells as stray bars,
#    so we use borders only, never background-color.
CSS = """
* { font-family: serif; }
body { font-size: 11pt; line-height: 1.4; color: #111111; }
h1 { font-size: 18pt; text-align: center; margin-bottom: 0pt; }
h2 { font-size: 13pt; margin-top: 14pt; margin-bottom: 4pt; }
h3 { font-size: 11.5pt; margin-top: 10pt; }
p.center { text-align: center; margin: 0pt; }
p.title { font-size: 13pt; font-weight: bold; margin-bottom: 8pt; }
p { margin-top: 3pt; margin-bottom: 6pt; }
li { margin-bottom: 2pt; }
code { font-family: monospace; font-size: 10pt; }
blockquote { margin-left: 12pt; font-style: italic; }
table { border-spacing: 0; margin-top: 4pt; margin-bottom: 10pt; width: 100%; }
th { font-weight: bold; font-size: 10pt; padding: 4pt; text-align: left; border-top: 1.5px solid #111111;
     border-bottom: 1.5px solid #111111; }
td { border-bottom: 0.5px solid #777777; font-size: 10pt; padding: 4pt; vertical-align: top; }
div.box { border: 1.2px solid #111111; padding: 8pt; margin-top: 6pt; margin-bottom: 10pt; }
div.box p { margin-top: 0pt; }
hr { margin-top: 8pt; margin-bottom: 8pt; }
"""

md = MarkdownIt("commonmark", {"breaks": True}).enable("table")

DOCUMENTS = ["question_paper", "marking_scheme", "methodology"]


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Read the small `--- key: value ---` header at the top of each source file."""
    meta: dict[str, str] = {}
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"')
        text = text[match.end() :]
    return meta, text


def render_pdf(source: Path) -> Path:
    meta, body = split_front_matter(source.read_text(encoding="utf-8"))
    html = f"<body>{md.render(body)}</body>"
    target = OUTPUT / f"{source.stem}.pdf"

    # 1) Lay the HTML out across as many A4 pages as it needs
    story = pymupdf.Story(html=html, user_css=CSS)
    writer = pymupdf.DocumentWriter(str(target))
    more = True
    while more:
        device = writer.begin_page(PAGE)
        more, _ = story.place(PAGE + MARGIN)
        story.draw(device)
        writer.end_page()
    writer.close()

    # 2) Running header, page numbers and metadata, like a real exam paper
    doc = pymupdf.open(target)
    total = doc.page_count
    grey = (0.35, 0.35, 0.35)
    for number, page in enumerate(doc, start=1):
        page.insert_text((56, 34), meta.get("header", ""), fontsize=8, fontname="tiro", color=grey)
        page.insert_text((56, PAGE.height - 30), "Northbridge Institute · CS201-MT-2026", fontsize=8, color=grey)
        footer = f"Page {number} of {total}" + ("   (turn over)" if number < total else "")
        page.insert_text((PAGE.width - 160, PAGE.height - 30), footer, fontsize=8, color=grey)
    doc.set_metadata(
        {
            "title": f"CS201 {meta.get('title', source.stem)}",
            "author": "Dr. Emma Carter (fictional), for the GradeWise course",
            "subject": "CS201 Systems Fundamentals midterm",
            "keywords": "CS201; midterm; Northbridge Institute; GradeWise",
            "creator": "scripts/build_exam_pack.py",
        }
    )
    doc.saveIncr()
    doc.close()
    return target


def check_against_expected() -> dict[str, str]:
    """💡 The PDFs are for humans and LLMs; expected_exam.json is the answer key. Make sure they agree."""
    if not EXPECTED.exists():
        raise DataNotFoundError(f"{EXPECTED.name} is missing from {OUTPUT}.")
    exam = ExamPaper.model_validate_json(EXPECTED.read_text(encoding="utf-8"))
    if exam.problems():
        raise ValueError(f"expected_exam.json is inconsistent: {exam.problems()}")

    # Marks printed on the question paper, in order, must match max_marks in the JSON
    paper = (SOURCE / "question_paper.md").read_text(encoding="utf-8")
    printed = [float(m) for m in re.findall(r"\[(\d+(?:\.\d+)?)\s+marks?\]", paper.split("---", 2)[2])]
    printed = printed[1:]  # ⚠️ the first "[4 marks]" is the example in the instructions box
    expected = [q.max_marks for q in exam.questions]
    if printed != expected:
        raise ValueError(f"Question paper marks {printed} don't match expected_exam.json {expected}")

    # Every question has a model answer section
    answers = set(re.findall(r"^## (Q\w+)\s*$", MODEL_ANSWERS.read_text(encoding="utf-8"), re.M))
    missing = [q.id for q in exam.questions if q.id not in answers]
    if missing:
        warn(f"model_answers.md has no section for {missing}")
    return {
        "Questions (incl. sub-parts)": str(len(exam.questions)),
        "Criteria": str(sum(len(q.criteria) for q in exam.questions)),
        "Total marks": f"{exam.questions_total:g}",
    }


if __name__ == "__main__":
    banner("Building the CS201 midterm exam pack", "Markdown → question paper, marking scheme, methodology PDFs")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    built: dict[str, str] = {}
    for name in DOCUMENTS:
        source = SOURCE / f"{name}.md"
        step(f"Rendering {source.name}")
        pdf = render_pdf(source)
        with pymupdf.open(pdf) as doc:
            built[pdf.name] = f"{doc.page_count} pages"
    step("Checking the PDFs agree with expected_exam.json")
    built |= check_against_expected()
    success("Exam pack built", built, next_step="week 1 day 5 extracts these PDFs into an ExamPaper")
