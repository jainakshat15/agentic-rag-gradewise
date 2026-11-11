"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 2 · Day 4: Dense, BM25 and hybrid retrieval                 │
│                                                                      │
│  What this file does                                                 │
│    dense  → embeddings + Chroma: finds chunks that MEAN the same     │
│    bm25   → keyword scoring:     finds chunks that SAY the same      │
│    hybrid → both, merged with Reciprocal Rank Fusion (RRF)           │
│    All three accept metadata filters, e.g. {"unit": 1}.              │
│                                                                      │
│  Use it                                                              │
│    from gradewise.rag import get_retriever                           │
│    docs = get_retriever(k=5).invoke("four conditions for deadlock")  │
│                                                                      │
│  ⏮ Built on: gradewise/ingestion   ⏭ Used by: answer.py, evidence.py│
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from gradewise.errors import GradeWiseError
from gradewise.ingestion.store import DEFAULT_COLLECTION, get_vectorstore

Mode = Literal["dense", "bm25", "hybrid"]
RRF_K = 60  # 💡 the constant from the original RRF paper (Cormack et al., 2009); rarely worth tuning


class EmptyKnowledgeBaseError(GradeWiseError):
    title = "The knowledge base is empty"
    cause = "Nothing has been ingested into this collection yet (or OUTPUT_DIR points somewhere new)."
    fix = "Run `uv run week2/pro_implementation/ingest.py` (or `uv run gradewise ingest`) first."
    help_ref = "week2/day5.ipynb"


# ── Helpers ──────────────────────────────────────────────────────────────


def doc_key(doc: Document) -> str:
    """A stable identity for a chunk, used to merge the same chunk found by two retrievers."""
    meta = doc.metadata or {}
    if "source" in meta and "chunk_index" in meta:
        return f"{meta['source']}#{meta['chunk_index']}"
    return doc.id or str(hash(doc.page_content))


def to_chroma_where(filter: dict[str, Any] | None) -> dict[str, Any] | None:
    """{"unit": 1, "source": "x.pdf"} → Chroma's {"$and": [{"unit": 1}, {"source": "x.pdf"}]}.

    ⚠️ Chroma only accepts ONE key at the top level of `where`; several conditions must be
       wrapped in "$and". Filters that already use operators ("$and", "$or") pass through.
    """
    if not filter:
        return None
    if len(filter) == 1 or any(key.startswith("$") for key in filter):
        return filter
    return {"$and": [{key: value} for key, value in filter.items()]}


# 💡 (ending, replacement), longest first: "reliability" and "reliable" both become "reli",
#    "isolation" and "isolated" both become "isolat".
_SUFFIXES = (
    ("ibility", ""),
    ("ability", ""),
    ("ation", "at"),
    ("ible", ""),
    ("able", ""),
    ("ity", ""),
    ("ing", ""),
    ("ed", ""),
    ("ly", ""),
)
_VOWEL = re.compile(r"[aeiouy]")


def stem(word: str) -> str:
    """A tiny, dependency-free "Porter-lite" stemmer, so BM25 matches word families.

    reliable / reliability → reli · processes / process → process · policies → policy
    scheduling / schedule → schedul · committed → commit · locks → lock

    ⚠️ It's deliberately crude: it only has to map related words to the SAME string, not to a
       real word. Short words and tokens with digits ('ipv4', '2nf', '802') are left alone.
    🚀 PRO: for a real system, try a proper stemmer (nltk's Porter / Snowball) or lemmatiser.
    """
    if len(word) <= 3 or not word.isalpha():
        return word
    # 1️⃣ Plurals: policies → policy, classes → class, locks → lock (but not "status", "analysis", "access")
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    # 2️⃣ One derivational / inflectional ending, if a real stem (≥ 3 letters with a vowel) is left
    for suffix, replacement in _SUFFIXES:
        base = word[: -len(suffix)]
        if word.endswith(suffix) and len(base) >= 3 and _VOWEL.search(base):
            word = base + replacement
            if suffix in ("ing", "ed") and len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsz":
                word = word[:-1]  # running → run, committed → commit
            break
    # 3️⃣ A final silent 'e': schedule → schedul, cache → cach (to meet "scheduling", "caching")
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def tokenize(text: str, *, stemming: bool = True) -> list[str]:
    """Lower-case word tokens, stemmed. Keeps things like 'ipv4', '2nf' and 'tcp' intact.

    💡 The SAME function tokenises the chunks and the query, so "reliable" in a question
       finds "reliability" in the notes. Pass `stemming=False` to see the raw tokens.
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(t) for t in tokens] if stemming else tokens


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]],
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
) -> list[Document]:
    """Merge several ranked lists into one: score(d) = Σ weight / (k + rank of d in each list).

    💡 RRF only looks at RANKS, never raw scores. That's why it can fuse a cosine distance
       (0…2, lower is better) with a BM25 score (0…∞, higher is better) without any scaling.
    """
    weights = list(weights) if weights is not None else [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    first_seen: dict[str, Document] = {}
    for docs, weight in zip(ranked_lists, weights, strict=True):
        for rank, doc in enumerate(docs, start=1):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            first_seen.setdefault(key, doc)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [first_seen[key] for key in ordered]


# ── BM25 over the chunks already stored in Chroma ────────────────────────

_BM25_CACHE: dict[tuple, tuple[Any, list[Document]]] = {}


def _bm25_index(store, where: dict[str, Any] | None):
    """Build (or reuse) a BM25 index over the stored chunks that match `where`."""
    from rank_bm25 import BM25Okapi

    count = store._collection.count()
    cache_key = (id(store._client), store._collection.name, json.dumps(where, sort_keys=True), count)
    if cache_key not in _BM25_CACHE:
        data = store.get(where=where, include=["documents", "metadatas"])
        docs = [
            Document(page_content=text, metadata=meta or {}, id=chunk_id)
            for chunk_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"], strict=True)
        ]
        # 💡 The index is rebuilt only when the number of chunks changes (e.g. after ingest).
        index = BM25Okapi([tokenize(d.page_content) for d in docs]) if docs else None
        _BM25_CACHE[cache_key] = (index, docs)
    return _BM25_CACHE[cache_key]


# ── The retriever ────────────────────────────────────────────────────────


class GradeWiseRetriever(BaseRetriever):
    """A LangChain retriever over a GradeWise Chroma collection: dense, BM25 or hybrid."""

    store: Any
    k: int = 5
    mode: Mode = "hybrid"
    filter: dict[str, Any] | None = None
    fetch_k: int = 20  # 💡 candidates taken from EACH retriever before fusion
    weights: tuple[float, float] = (1.0, 1.0)  # (dense, bm25) weights for RRF

    def dense(self, query: str, k: int | None = None) -> list[Document]:
        return self.store.similarity_search(query, k=k or self.k, filter=to_chroma_where(self.filter))

    def bm25(self, query: str, k: int | None = None) -> list[Document]:
        index, docs = _bm25_index(self.store, to_chroma_where(self.filter))
        if index is None:
            return []
        scores = index.get_scores(tokenize(query))
        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        # ⚠️ A score of 0 means "no query word appears at all": not a match, just noise.
        return [docs[i] for i in ranked[: k or self.k] if scores[i] > 0]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        if self.store._collection.count() == 0:
            raise EmptyKnowledgeBaseError(f"The collection '{self.store._collection.name}' has no chunks.")
        if self.mode == "dense":
            return self.dense(query)
        if self.mode == "bm25":
            return self.bm25(query)
        fused = reciprocal_rank_fusion(
            [self.dense(query, self.fetch_k), self.bm25(query, self.fetch_k)], weights=self.weights
        )
        return fused[: self.k]


def get_retriever(
    collection: str = DEFAULT_COLLECTION,
    k: int = 5,
    mode: Mode = "hybrid",
    filter: dict[str, Any] | None = None,
    embeddings: Any | None = None,
    persist_dir: Path | str | None = None,
    store: Any | None = None,
) -> GradeWiseRetriever:
    """The one way GradeWise retrieves from its knowledge base.

    Args:
        collection: which course's knowledge base ("cs201").
        k: how many chunks to return.
        mode: "dense", "bm25" or "hybrid" (the default, usually the best).
        filter: metadata filter, e.g. {"unit": 1} or {"source": "unit3_databases.pdf"}.
        embeddings / persist_dir / store: injectable, mainly for tests.

    🧪 TRY: run week2/evaluation/eval.py with --pipeline dense, bm25 and pro (hybrid)
       and see which wins on each question category.
    """
    if mode not in ("dense", "bm25", "hybrid"):
        raise ValueError(f"mode must be 'dense', 'bm25' or 'hybrid', not {mode!r}")
    store = store or get_vectorstore(collection, embeddings=embeddings, persist_dir=persist_dir)
    return GradeWiseRetriever(store=store, k=k, mode=mode, filter=filter)
