"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 1 · Day 5: 🧱 PROJECT: the exam pack → ExamPaper JSON (v1)  │
│                                                                      │
│  What this script does                                               │
│    Reads Emma's three PDFs (question paper, marking scheme,          │
│    methodology), asks the model for ONE structured ExamPaper, and    │
│    saves it as output/cs_midterm.json. One call, no graph: simple.   │
│                                                                      │
│  Run:  uv run week1/project/extract_exam.py                          │
│                                                                      │
│  ⏮ Built on: week1/day3.ipynb (structured output), day4 (PDFs)       │
│  ⏭ Used by: week3/project/exam_graph.py (v2 adds retry + approval)   │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf4llm
from rich.table import Table

from gradewise.config import DATA_DIR, OUTPUT_DIR
from gradewise.console import banner, console, step, success, warn
from gradewise.errors import DataNotFoundError, GradeWiseError
from gradewise.models import chat_model
from gradewise.schemas import ExamPaper

EXAM_ID = "cs_midterm"
EXAM_DIR = DATA_DIR / "exams" / EXAM_ID
FILES = ["question_paper.pdf", "marking_scheme.pdf", "methodology.pdf"]

# 💡 The system prompt is a checklist. Each rule fixes a mistake models really make.
SYSTEM_PROMPT = """\
You turn an exam pack into ONE ExamPaper: a single array of questions with their marking criteria and method.
Rules:
1. One entry per separately-marked sub-part ("Q7a", "Q7b"), never the parent as well. Unsplit questions are "Q3".
2. Criterion IDs are the question ID + ".c" + a counter: "Q7a.c1", "Q7a.c2".
3. Copy the question text exactly as printed. For a sub-part, start with the parent's shared stem.
4. NEVER invent marks. Every mark comes from the documents. Criteria must add up to the question's marks,
   and questions to the paper total.
5. Every question gets a `methodology`: the rules for its type of question, anything said about it
   specifically, and the general marking principles that always apply.
6. `model_answer` is copied from the MODEL ANSWERS section when given, otherwise null.
"""


def read_pdf(path: Path) -> str:
    """A PDF as Markdown. pymupdf4llm keeps headings, bold and tables, which LLMs read well."""
    if not path.exists():
        raise DataNotFoundError(f"Can't find {path}.")
    return pymupdf4llm.to_markdown(str(path), show_progress=False)


def build_prompt() -> str:
    """All the documents in one message, each under a clear heading."""
    parts = []
    for name in FILES:
        step(f"Reading {name}…")
        parts.append(f"=== {name.removesuffix('.pdf').replace('_', ' ').upper()} ===\n{read_pdf(EXAM_DIR / name)}")
    answers = EXAM_DIR / "model_answers.md"
    if answers.exists():  # 💡 optional extra: the examiner's model answers
        parts.append(f"=== MODEL ANSWERS ===\n{answers.read_text(encoding='utf-8')}")
    return f'Exam ID: "{EXAM_ID}"\n\n' + "\n\n".join(parts)


def show(paper: ExamPaper) -> None:
    """A quick table so you can eyeball the result."""
    table = Table(title=paper.title, header_style="accent")
    for column in ("ID", "Type", "Marks", "Criteria"):
        table.add_column(column)
    for q in paper.questions:
        table.add_row(q.id, q.question_type, f"{q.max_marks:g}", str(len(q.criteria)))
    console.print(table)


def main() -> int:
    banner("Week 1 · Day 5", "Exam pack → ExamPaper JSON")

    # 💡 Create the model first: if your key is missing, you find out before any work is done.
    llm = chat_model(temperature=0)
    # 💡 The one line that matters: the model must answer in the shape of ExamPaper.
    extractor = llm.with_structured_output(ExamPaper)

    prompt = build_prompt()
    step("Asking the model for the ExamPaper (this takes a few seconds)…")
    paper = extractor.invoke([("system", SYSTEM_PROMPT), ("human", prompt)])
    paper = paper.model_copy(update={"exam_id": EXAM_ID})  # ⚠️ we choose the ID, not the model

    # 💡 Pydantic already checked the *shape*. problems() checks the *arithmetic*.
    for problem in paper.problems():
        warn(problem)
    # 🧪 TRY: delete rule 4 from the prompt and run again. Do the marks still add up?

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{EXAM_ID}.json"
    out.write_text(json.dumps(paper.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

    show(paper)
    success(
        "Exam extracted",
        {
            "Questions": len(paper.questions),
            "Marks": f"{paper.questions_total:g} / {paper.total_marks:g}",
            "Problems": len(paper.problems()) or "none 🎉",
            "Saved to": out,
        },
        next_step="Week 2: build the knowledge base GradeWise grades against",
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GradeWiseError as err:  # 💡 missing key, missing files...: a friendly red panel, not a traceback
        err.show()
        sys.exit(1)
