"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🚢 Week 6 · Day 1: Demo mode, the whole app with no API key         │
│                                                                      │
│  What this file does                                                 │
│    Offline stand-ins for every model the API would call, so you can  │
│    click through the full product (upload → grade → review → report  │
│    → chat) for free, on a plane, or in a CI job:                     │
│                                                                      │
│      DemoExamModel      → "extracts" the bundled CS201 midterm       │
│      DemoGraderModel    → week 5's KeywordGraderModel, a little      │
│                            unsure about low marks (so the review     │
│                            queue has something in it)                │
│      DemoResultsModel   → picks one Q&A tool and shows its output    │
│      DemoRetriever      → BM25 over the demo knowledge base          │
│      demo_embeddings()  → DeterministicFakeEmbedding                 │
│                                                                      │
│  ⚠️ The marks mean NOTHING. Demo mode exercises the plumbing (API,   │
│     background jobs, interrupts, SSE, storage), not the judgement.   │
│  ⏮ Built on: gradewise/grading/fakes.py                             │
│  ⏭ Used by: main.py, jobs.py, tests/test_api.py                     │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from gradewise.config import DATA_DIR
from gradewise.grading.fakes import KeywordGraderModel, _tool_names, tool_call_message
from gradewise.schemas.answers import AnswerSheet, TranscribedAnswer
from gradewise.schemas.exam import ExamPaper

SAMPLE_EXAM = DATA_DIR / "exams" / "cs_midterm" / "expected_exam.json"
TRANSCRIPTS = DATA_DIR / "answer-sheets" / "synthetic" / "transcripts.json"
EMBEDDING_SIZE = 32


def demo_embeddings() -> DeterministicFakeEmbedding:
    """Same text → same vector, no network. ⚠️ Similar texts do NOT get similar vectors: dense search is noise."""
    return DeterministicFakeEmbedding(size=EMBEDDING_SIZE)


# ── 📝 Exam extraction ───────────────────────────────────────────────────────────
class DemoExamModel:
    """Quacks like a chat model for `with_structured_output(ExamPaper)`: always returns the sample exam.

    💡 Whatever PDFs you upload, the demo "extracts" data/exams/cs_midterm/expected_exam.json. The parse,
       validate, interrupt-for-approval and store steps around it are all the real thing.
    """

    def with_structured_output(self, schema, **kwargs):  # noqa: ARG002
        def respond(_messages):
            return schema.model_validate_json(SAMPLE_EXAM.read_text(encoding="utf-8"))

        return RunnableLambda(respond)


# ── 🧮 Grading ───────────────────────────────────────────────────────────────────
class DemoGraderModel(KeywordGraderModel):
    """The week 5 keyword examiner, but it reports LOW confidence when it awards half marks or less.

    💡 Low confidence → guardrails flag the answer → the batch pauses at human_review. That gives the
       review queue real work to show, exactly like a cautious real grader would.
    """

    unsure_confidence: float = 0.55

    @property
    def _llm_type(self) -> str:
        return "demo-grader-fake"

    def _respond(self, messages: list[BaseMessage]) -> AIMessage:
        message = super()._respond(messages)
        call = message.tool_calls[0] if message.tool_calls else None
        if call and call["name"] == "GradeResult":
            args = dict(call["args"])
            share = args["awarded_marks"] / max(float(args["max_marks"]), 1e-9)
            if share <= 0.5:
                args["confidence"] = self.unsure_confidence
            return tool_call_message("GradeResult", args, call["id"])
        return message


class DemoRetriever(BaseRetriever):
    """BM25 over the demo knowledge base; returns nothing (instead of raising) while it's still empty."""

    collection: str = "cs201"
    persist_dir: str
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:  # noqa: ARG002
        from gradewise.rag.retrievers import GradeWiseRetriever

        store = _demo_store(self.collection, self.persist_dir)
        if store._collection.count() == 0:
            return []  # 💡 nothing ingested yet: grade without evidence rather than crash the demo
        # 💡 mode="bm25": keyword search is meaningful even with fake embeddings (dense search isn't).
        return GradeWiseRetriever(store=store, k=self.k, mode="bm25").invoke(query)


@lru_cache(maxsize=8)
def _demo_store(collection: str, persist_dir: str):
    """One Chroma handle per (collection, folder), reused by every question of every batch."""
    from gradewise.ingestion.store import get_vectorstore

    return get_vectorstore(collection, embeddings=demo_embeddings(), persist_dir=persist_dir)


# ── ✍️ Transcription ─────────────────────────────────────────────────────────────
def load_transcripts() -> dict[str, dict[str, str]]:
    if not TRANSCRIPTS.exists():
        return {}
    return json.loads(TRANSCRIPTS.read_text(encoding="utf-8"))


def demo_sheet(exam: ExamPaper, pdf_path: str | Path) -> AnswerSheet:
    """Stand in for the vision model: a synthetic student's known transcript, matched by file name.

    - student_04.pdf → the answers from transcripts.json (confidence 0.95);
    - any other PDF → blank answers with confidence 0, so the batch pauses at review_transcription
      and you can type in what the student wrote: the human-in-the-loop path, for free.
    """
    path = Path(pdf_path)
    truth = load_transcripts().get(path.stem)
    answers = [
        TranscribedAnswer(
            question_id=q.id,
            text=(truth or {}).get(q.id, ""),
            confidence=0.95 if truth else 0.0,
            illegible_spans=[] if truth else ["(demo mode can't read handwriting)"],
        )
        for q in exam.questions
    ]
    return AnswerSheet(student_id=path.stem, exam_id=exam.exam_id, source_file=path.name, page_count=0, answers=answers)


# ── 💬 Chat with results ─────────────────────────────────────────────────────────
class DemoResultsModel(BaseChatModel):
    """An offline "results assistant": picks ONE of the four Q&A tools by keyword, then shows its output.

    Real models read the tool output and write a sentence or two; this one is honest about being a fake.
    """

    bound_tools: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "demo-results-fake"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ARG002
        return self.model_copy(update={"bound_tools": _tool_names(tools)})

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ARG002
        question = next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        tool_results = [m for m in messages if isinstance(m, ToolMessage)]
        if not tool_results:
            name, args = self.choose_tool(question)
            return ChatResult(generations=[ChatGeneration(message=tool_call_message(name, args))])
        last = tool_results[-1]
        text = (
            f"🧪 **Demo mode**: no language model is answering. I called `{last.name}` "
            f"and here is exactly what it returned:\n\n```text\n{last.content}\n```"
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @staticmethod
    def choose_tool(question: str) -> tuple[str, dict[str, Any]]:
        q = question.lower()
        student = re.search(r"student_\d+", q)
        qid = re.search(r"\bq\d+[a-z]?\b", q)
        if any(w in q for w in ("flag", "review", "pending", "check")):
            return "list_flagged", {}
        if student:
            return "get_student_report", {"student_id": student.group(0)}
        if any(w in q for w in ("which question", "hardest", "easiest", "average", "mean")):
            return "class_statistics", {"question_id": "Q" + qid.group(0)[1:] if qid else ""}
        if any(w in q for w in ("who ", "which student", "struggl", "confus", "misunderst", "trouble")):
            return "search_reports", {"query": question[:200]}
        return "class_statistics", {"question_id": "Q" + qid.group(0)[1:] if qid else ""}
