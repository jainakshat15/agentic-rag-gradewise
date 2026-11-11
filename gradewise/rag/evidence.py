"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 4 · Day 2: The GradeWise evidence retriever                 │
│                                                                      │
│  What this file does                                                 │
│    For one exam question, finds the passages of the course notes a   │
│    marker would check the answer against. It searches with the       │
│    question, its reference topics and each marking criterion, then  │
│    fuses the results (RRF) into one short, de-duplicated list.       │
│                                                                      │
│  Use it                                                              │
│    from gradewise.rag import find_evidence                           │
│    docs = find_evidence(question, answer_text, k=4)                  │
│                                                                      │
│  Run : uv run week4/project/evidence.py                              │
│  ⏮ Built on: retrievers.py   ⏭ Used by: week5 grading agent         │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from gradewise.rag.citations import format_citation
from gradewise.rag.retrievers import doc_key, get_retriever, reciprocal_rank_fusion
from gradewise.schemas import Question

ANSWER_WEIGHT = 0.3  # 💡 the student's answer counts, but much less than the examiner's text


def evidence_queries(question: Question) -> list[str]:
    """The search queries for one question, built ONLY from what the examiner wrote."""
    queries = [question.text, *question.reference_topics]
    for criterion in question.criteria:
        keywords = f" ({', '.join(criterion.keywords)})" if criterion.keywords else ""
        queries.append(f"{criterion.description}{keywords}")
    return list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))


def find_evidence(
    question: Question,
    answer_text: str,
    k: int = 4,
    retriever: Any | None = None,
    collection: str = "cs201",
) -> list[Document]:
    """Return the `k` best passages of the course notes for marking `question`.

    Each returned Document gets two extra metadata keys:
      - `citation`: e.g. "unit1_operating_systems.pdf p4 §1.6 Deadlocks"
      - `matched_queries`: which of our queries found it (handy when explaining a grade)

    ⚠️ Why we DON'T search with the student's answer alone:
       a wrong answer pulls in evidence for the wrong idea. If Sam writes "a deadlock is when
       one process uses all the CPU", searching his words returns the CPU-scheduling notes,
       and the grader would then see "evidence" that seems to support him. The evidence must
       describe what a CORRECT answer contains, so it comes from the question, its reference
       topics and the marking criteria: the examiner's words, not the student's.
       We still add the answer as one LOW-weight query, so a correct-but-unexpected point
       (e.g. a valid example the scheme didn't list) can find its supporting passage too.
    """
    retriever = retriever or get_retriever(collection=collection, k=max(k, 5))
    queries = evidence_queries(question)

    ranked_lists = [retriever.invoke(q) for q in queries]
    weights = [1.0] * len(ranked_lists)
    if answer_text and answer_text.strip():
        ranked_lists.append(retriever.invoke(answer_text.strip()[:1000]))
        weights.append(ANSWER_WEIGHT)

    fused = reciprocal_rank_fusion(ranked_lists, weights=weights)[:k]

    results = []
    for doc in fused:
        key = doc_key(doc)
        matched = [q for q, docs in zip(queries, ranked_lists, strict=False) if any(doc_key(d) == key for d in docs)]
        meta = {**doc.metadata, "citation": format_citation(doc), "matched_queries": matched}
        results.append(Document(page_content=doc.page_content, metadata=meta, id=doc.id))
    return results
