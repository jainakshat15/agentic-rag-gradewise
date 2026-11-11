"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 2 · Day 5: Answering, the PRO version                       │
│                                                                      │
│  What this file does                                                 │
│    The same job as ../implementation/answer.py, with the pro         │
│    choices from the gradewise package:                               │
│      🚀 hybrid retrieval: dense + BM25, fused with RRF               │
│      🚀 optional metadata filters, e.g. only Unit 2                  │
│      🚀 citations like [unit1_operating_systems.pdf p4 §1.6 ...]     │
│      🚀 a prompt that treats retrieved text as data, not orders      │
│                                                                      │
│  Run : uv run week2/pro_implementation/answer.py "What is TCP?"      │
│  ⏮ Built on: gradewise/rag   ⏭ Used by: week2/evaluation/eval.py    │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.documents import Document  # noqa: E402

from gradewise.console import console, success  # noqa: E402
from gradewise.errors import GradeWiseError  # noqa: E402
from gradewise.rag import answer_question as rag_answer  # noqa: E402
from gradewise.rag import get_retriever, unique_citations  # noqa: E402

RETRIEVAL_K = 5


def fetch_context(question: str, k: int = RETRIEVAL_K, mode: str = "hybrid", unit: int | None = None) -> list[Document]:
    """Hybrid retrieval, optionally restricted to one unit of the course."""
    # 🧪 TRY: unit=2 for a networks-only assistant, or mode="dense" to see what BM25 adds.
    return get_retriever(k=k, mode=mode, filter={"unit": unit} if unit else None).invoke(question)


def answer_question(question: str, llm=None) -> tuple[str, list[Document]]:
    """Retrieve (hybrid), then generate an answer with citations."""
    return rag_answer(question, k=RETRIEVAL_K, llm=llm)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What are the four necessary conditions for deadlock?"
    try:
        answer, docs = answer_question(question)
    except GradeWiseError as err:
        err.show()
        sys.exit(1)
    console.print(f"[primary]Q:[/] {question}\n\n{answer}\n")
    success("Answered (pro RAG)", {"Sources": "\n".join(unique_citations(docs))})
