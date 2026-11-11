"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Weeks 2 and 4: Retrieval and answering                           │
│                                                                      │
│  What this package does                                              │
│      retrievers.py   → dense, BM25, hybrid (RRF) + filters  (wk2 d4) │
│      citations.py    → "unit1_….pdf p4 §1.6 Deadlocks"      (wk2 d4) │
│      answer.py       → 2-step RAG answer with citations     (wk2 d4) │
│      agentic_rag.py  → LangGraph retrieve/grade/rewrite     (wk4 d1) │
│      evidence.py     → evidence for grading one question    (wk4 d2) │
│                                                                      │
│  ⏮ Built on: gradewise/ingestion   ⏭ Used by: week5 grading         │
╰──────────────────────────────────────────────────────────────────────╯
"""

from gradewise.rag.answer import answer_question
from gradewise.rag.citations import format_citation, format_context, unique_citations
from gradewise.rag.retrievers import (
    EmptyKnowledgeBaseError,
    GradeWiseRetriever,
    get_retriever,
    reciprocal_rank_fusion,
)

__all__ = [
    "EmptyKnowledgeBaseError",
    "GradeWiseRetriever",
    "answer_question",
    "build_agentic_rag",
    "find_evidence",
    "format_citation",
    "format_context",
    "get_retriever",
    "reciprocal_rank_fusion",
    "run_agentic_rag",
    "unique_citations",
]


def __getattr__(name: str):
    # 💡 Lazy imports: `from gradewise.rag import get_retriever` shouldn't have to load
    #    LangGraph (agentic_rag) or the exam schemas (evidence) if you don't use them.
    if name in ("build_agentic_rag", "run_agentic_rag"):
        from gradewise.rag import agentic_rag

        return getattr(agentic_rag, name)
    if name == "find_evidence":
        from gradewise.rag.evidence import find_evidence

        return find_evidence
    raise AttributeError(f"module 'gradewise.rag' has no attribute {name!r}")
