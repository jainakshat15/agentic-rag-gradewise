"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 1 · Day 5 (v1) + Week 3 · Day 5 (self-correction)           │
│                                                                      │
│  What this file does                                                 │
│    Three documents in, ONE ExamPaper out. The question paper, the    │
│    marking scheme and the methodology are joined into a single       │
│    array of questions, each with its criteria and marking method.    │
│                                                                      │
│      extract_exam(...)        → one structured-output call (wk 1)    │
│      extract_with_retry(...)  → checks problems() and sends them     │
│                                 back so the model fixes itself (wk 3)│
│                                                                      │
│  Use it                                                              │
│    from gradewise.exam.extract import extract_with_retry             │
│    paper = extract_with_retry(qp_pdf, scheme_pdf, method_pdf)        │
│                                                                      │
│  ⏮ Built on: week1/day3.ipynb (structured output), schemas/exam.py   │
│  ⏭ Used by: gradewise/exam/graph.py, week1/project, week3/project    │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from gradewise.config import DATA_DIR
from gradewise.console import step, warn
from gradewise.errors import DataNotFoundError, GradeWiseError
from gradewise.exam.parse import ParsedDocument, read_document
from gradewise.schemas.exam import ExamPaper

Source = str | Path | ParsedDocument

# 💡 The three files every exam pack contains, in the order we show them to the model.
PACK_FILES = {
    "question_paper": "question_paper.pdf",
    "marking_scheme": "marking_scheme.pdf",
    "methodology": "methodology.pdf",
}
# 💡 Optional: the examiner's model answers. Used for `model_answer` when present.
MODEL_ANSWERS_FILE = "model_answers.md"


class ExamExtractionError(GradeWiseError):
    title = "Couldn't extract the exam"
    cause = "The model's answer never matched the ExamPaper schema, even after retries."
    fix = "Check the PDFs read cleanly (gradewise.exam.parse), try a stronger CHAT_MODEL, or raise max_attempts."
    help_ref = "week3/project/README.md"


# 💡 This prompt is the heart of week 1 day 5. Every rule here exists because a model
#    got it wrong at least once: read it slowly, it's a checklist of extraction pitfalls.
SYSTEM_PROMPT = """\
You are GradeWise's exam-understanding assistant at Northbridge Institute.
You receive three documents for ONE exam: the QUESTION PAPER, the MARKING SCHEME and the MARKING METHODOLOGY.
Join them into a single ExamPaper: one array of questions, where each question carries its own marking criteria
and its own marking method.

Rules:
1. One entry per markable unit. If a question has sub-parts that are marked separately (4(a), 4(b)), emit one
   question per sub-part and do NOT also emit the parent. A question with no sub-parts is a single entry.
2. Stable IDs. Question IDs are "Q" + number + optional sub-part letter, with no spaces or brackets:
   "Q3", "Q7a", "Q7b". Criterion IDs are the question ID + ".c" + a 1-based counter: "Q7a.c1", "Q7a.c2".
   `number` is the label exactly as printed, e.g. "7(a)".
3. Copy, don't paraphrase. `text` is the full question text exactly as printed, including any data the student
   needs (sizes, addresses, tables). For a sub-part, start with the parent question's shared stem, so each entry
   reads on its own: "A server has the IP address 172.16.45.200/27. Find the network address...".
   Criterion descriptions follow the marking scheme's wording.
4. NEVER invent marks. Every number in `max_marks`, `total_marks` and each criterion's `marks` must come from the
   documents. A question's criteria must add up to its printed marks, and the questions must add up to the paper
   total. If they don't, re-read the marking scheme: you have probably missed or merged a criterion.
5. Methodology goes on every question, so a grader never needs the other documents. Into `methodology` copy:
   the rules for this TYPE of question (e.g. how numericals or definitions are marked), then anything said about
   THIS question specifically, then the general principles that always apply (spelling tolerance, positive
   marking, never exceed the maximum, ...). Never leave it empty.
   Put criterion-specific partial-credit rules in that criterion's `partial_credit_rule`.
6. `keywords` are helpful terms an examiner looks for (from the scheme), never hard requirements.
7. `question_type` is one of: definition, short, long, compare, numerical, mcq. Pick the closest fit.
8. `reference_topics` are 1-4 short course topics a grader should look up in the CS201 notes,
   e.g. ["deadlock conditions"].
9. `model_answer` is copied from the MODEL ANSWERS document when one is given for that question; otherwise null.
   Never write your own.
10. `title` and `subject` come from the question paper's heading.
"""


def build_messages(
    question_paper: str,
    marking_scheme: str,
    methodology: str,
    *,
    exam_id: str = "cs_midterm",
    model_answers: str | None = None,
    previous: ExamPaper | dict[str, Any] | None = None,
    feedback: list[str] | None = None,
) -> list[BaseMessage]:
    """The chat messages for one extraction attempt.

    On a retry we add the model's previous answer AND the list of problems, so it can fix
    its own work instead of starting from scratch: this is self-correction.
    """
    documents = (
        f'Exam ID: "{exam_id}"\n\n'
        f"=== QUESTION PAPER ===\n{question_paper.strip()}\n\n"
        f"=== MARKING SCHEME ===\n{marking_scheme.strip()}\n\n"
        f"=== MARKING METHODOLOGY ===\n{methodology.strip()}"
    )
    if model_answers:
        documents += f"\n\n=== MODEL ANSWERS ===\n{model_answers.strip()}"
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(documents)]
    if feedback:
        if previous is not None:
            data = previous.model_dump() if isinstance(previous, ExamPaper) else previous
            messages.append(AIMessage(json.dumps(data, indent=2, ensure_ascii=False)))
        issues = "\n".join(f"- {issue}" for issue in feedback)
        messages.append(
            HumanMessage(
                "Your previous extraction has these problems:\n"
                f"{issues}\n\n"
                "Re-read the documents and return the corrected, complete ExamPaper. Fix only what is wrong; keep "
                "the IDs of everything that was right. Do not invent or adjust marks to make totals add up: find the "
                "criterion or question you missed or misread."
            )
        )
    return messages


def extract_once(
    question_paper: Source,
    marking_scheme: Source,
    methodology: Source,
    *,
    exam_id: str = "cs_midterm",
    llm=None,
    model_answers: Source | None = None,
    previous: ExamPaper | dict[str, Any] | None = None,
    feedback: list[str] | None = None,
    schema: type[ExamPaper] = ExamPaper,
) -> tuple[ExamPaper | None, list[str]]:
    """One attempt. Returns `(paper, problems)`; `paper` is None if the output didn't even validate.

    💡 Both kinds of failure become a plain list of strings, which is exactly what we
       feed back to the model on the next attempt.
    💡 `schema` can be any ExamPaper subclass (e.g. an exercise's `MCQExamPaper`): the model
       is asked for that schema, and its own `problems()` checks drive the retries.
    """
    if llm is None:
        from gradewise.models import chat_model

        llm = chat_model(temperature=0)  # 💡 extraction wants the same answer every time

    messages = build_messages(
        read_document(question_paper),
        read_document(marking_scheme),
        read_document(methodology),
        exam_id=exam_id,
        model_answers=read_document(model_answers) if model_answers is not None else None,
        previous=previous,
        feedback=feedback,
    )
    structured = llm.with_structured_output(schema)
    try:
        result = structured.invoke(messages)
        # ⚠️ isinstance(…, schema), not ExamPaper: a plain ExamPaper is NOT a valid MCQExamPaper.
        paper = result if isinstance(result, schema) else schema.model_validate(result)
    except ValidationError as exc:
        return None, [f"Schema error at {'.'.join(map(str, e['loc'])) or 'root'}: {e['msg']}" for e in exc.errors()]
    except (OutputParserException, ValueError) as exc:
        return None, [f"The output could not be parsed as an {schema.__name__}: {exc}"]

    # ⚠️ We decide the exam ID, not the model: it's the key we store the exam under.
    paper = paper.model_copy(update={"exam_id": exam_id})
    return paper, paper.problems()


def extract_exam(
    question_paper: Source,
    marking_scheme: Source,
    methodology: Source,
    *,
    exam_id: str = "cs_midterm",
    llm=None,
    model_answers: Source | None = None,
    schema: type[ExamPaper] = ExamPaper,
) -> ExamPaper:
    """🔁 Week 1 v1: a single structured-output call. Raises if the output doesn't validate.

    `schema` may be an ExamPaper subclass; the returned paper is an instance of it.
    """
    paper, problems = extract_once(
        question_paper,
        marking_scheme,
        methodology,
        exam_id=exam_id,
        llm=llm,
        model_answers=model_answers,
        schema=schema,
    )
    if paper is None:
        raise ExamExtractionError(
            f"The model's answer didn't match the {schema.__name__} schema: " + "; ".join(problems)
        )
    for problem in problems:
        warn(problem)  # 💡 v1 just reports problems; extract_with_retry fixes them
    return paper


def extract_with_retry(
    question_paper: Source,
    marking_scheme: Source,
    methodology: Source,
    *,
    exam_id: str = "cs_midterm",
    llm=None,
    max_attempts: int = 3,
    model_answers: Source | None = None,
    schema: type[ExamPaper] = ExamPaper,
) -> ExamPaper:
    """Week 3: extract, check `problems()`, and send the issues back until they're gone.

    Returns the first clean paper. If problems remain after `max_attempts`, returns the best
    paper we got (fewest problems) with a warning: a human should review it.
    Raises ExamExtractionError if no attempt produced a valid ExamPaper at all.
    `schema` may be an ExamPaper subclass (its own `problems()` drives the retries).

    🚀 PRO: parse the PDFs once up front; retries then only pay for the LLM call.
    """
    if llm is None:
        from gradewise.models import chat_model

        llm = chat_model(temperature=0)
    texts = [read_document(s) for s in (question_paper, marking_scheme, methodology)]
    answers = read_document(model_answers) if model_answers is not None else None

    best: tuple[ExamPaper, list[str]] | None = None
    previous: ExamPaper | None = None
    feedback: list[str] | None = None
    for attempt in range(1, max_attempts + 1):
        step(f"Extraction attempt {attempt}/{max_attempts}…")
        paper, problems = extract_once(
            *texts,
            exam_id=exam_id,
            llm=llm,
            model_answers=answers,
            previous=previous,
            feedback=feedback,
            schema=schema,
        )
        if paper is not None and not problems:
            return paper
        for problem in problems:
            warn(problem)
        if paper is not None and (best is None or len(problems) < len(best[1])):
            best = (paper, problems)
        previous, feedback = (paper or previous), problems

    if best is None:
        raise ExamExtractionError(f"No valid {schema.__name__} after {max_attempts} attempts.")
    warn(f"{len(best[1])} problem(s) remain after {max_attempts} attempts. Ask a human to review this exam.")
    return best[0]


def exam_pack(exam_id: str = "cs_midterm", folder: str | Path | None = None) -> dict[str, Path]:
    """The files of an exam pack, e.g. data/exams/cs_midterm/. Raises if one of the three PDFs is missing.

    Keys: question_paper, marking_scheme, methodology, plus model_answers if that file exists.
    """
    folder = Path(folder) if folder else DATA_DIR / "exams" / exam_id
    paths = {key: folder / name for key, name in PACK_FILES.items()}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise DataNotFoundError(f"The exam pack in {folder} is missing: {', '.join(missing)}.")
    if (folder / MODEL_ANSWERS_FILE).exists():
        paths["model_answers"] = folder / MODEL_ANSWERS_FILE
    return paths
