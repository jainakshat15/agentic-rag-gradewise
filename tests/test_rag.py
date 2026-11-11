"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧪 tests/test_rag.py: ingestion, retrieval, metrics, agentic RAG    │
│                                                                      │
│  Run:  uv run python -m pytest tests/test_rag.py -q                  │
│  💡 Free to run: fake embeddings, fake chat models, a temp Chroma.   │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from gradewise.config import DATA_DIR, REPO_ROOT
from gradewise.ingestion import chunk_documents, get_vectorstore, ingest, load_pdf
from gradewise.ingestion.loaders import find_running_lines, strip_running_lines
from gradewise.rag import format_citation, get_retriever, reciprocal_rank_fusion
from gradewise.rag.retrievers import stem, to_chroma_where, tokenize
from gradewise.schemas import MarkingCriterion, Question

KB = DATA_DIR / "knowledge-base"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluation = _load(REPO_ROOT / "week2" / "evaluation" / "eval.py", "week2_eval_under_test")


def fake_embeddings():
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    """Ingest the real PDFs once into a temp Chroma, shared by the tests below."""
    db = tmp_path_factory.mktemp("vector_db")
    report = ingest(kb_dir=KB, collection="cs201_test", embeddings=fake_embeddings(), persist_dir=db, quiet=True)
    return db, report


# ── Ingestion ──────────────────────────────────────────────────────────


def test_ingest_real_pdfs(kb):
    db, report = kb
    assert report.files == 4
    assert report.pages >= 20
    assert report.chunks_added == report.chunks_total > 40
    store = get_vectorstore("cs201_test", embeddings=fake_embeddings(), persist_dir=db)
    metas = store.get(include=["metadatas"])["metadatas"]
    assert {m["source"] for m in metas} == {p.name for p in KB.glob("*.pdf")}
    for meta in metas:
        assert meta["page"] >= 1 and meta["title"] and meta["section"] and meta["file_hash"]
    assert {m["unit"] for m in metas} == {0, 1, 2, 3}
    assert any(m["section"] == "1.6 Deadlocks" for m in metas)


def test_second_ingest_adds_nothing(kb):
    db, first = kb
    again = ingest(kb_dir=KB, collection="cs201_test", embeddings=fake_embeddings(), persist_dir=db, quiet=True)
    assert again.chunks_added == 0
    assert len(again.skipped) == 4
    assert again.chunks_total == first.chunks_total


def test_changed_file_is_reindexed(tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    pdf = kb_dir / "unit3_databases.pdf"
    pdf.write_bytes((KB / "unit3_databases.pdf").read_bytes())
    db = tmp_path / "db"
    first = ingest(kb_dir=kb_dir, collection="cs201_change", embeddings=fake_embeddings(), persist_dir=db, quiet=True)
    pdf.write_bytes(pdf.read_bytes() + b"\n%changed\n")  # 💡 same content, different hash
    second = ingest(kb_dir=kb_dir, collection="cs201_change", embeddings=fake_embeddings(), persist_dir=db, quiet=True)
    assert second.updated == ["unit3_databases.pdf"]
    assert second.chunks_total == first.chunks_total  # old chunks were deleted, not duplicated


def test_headers_and_footers_are_stripped():
    pages = load_pdf(KB / "unit1_operating_systems.pdf")
    for page in pages:
        assert "Page " not in page.page_content or " of 7" not in page.page_content
        assert "CS201 Systems Fundamentals · CS201 Unit 1" not in page.page_content
        assert "Northbridge Institute · School of Computing" not in page.page_content
    assert [p.metadata["page"] for p in pages] == list(range(1, len(pages) + 1))


def test_running_line_detection_on_toy_pages():
    bodies = ["Alpha beta.", "Gamma delta.", "Epsilon zeta."]
    pages = [f"ACME notes\n# Topic {i}\n{body}\nACME Ltd Page {i} of 3" for i, body in enumerate(bodies, start=1)]
    running = find_running_lines(pages)
    cleaned = strip_running_lines(pages[1], running)
    assert cleaned == "# Topic 2\nGamma delta."


def test_chunks_keep_section_path():
    chunks = chunk_documents(load_pdf(KB / "unit2_computer_networks.pdf"))
    assert all(c.metadata["section_path"].startswith("CS201 Unit 2: Computer Networks") for c in chunks)
    assert any("2.5 The transport layer" in c.metadata["section"] for c in chunks)


def test_tiny_chunks_are_merged_into_their_section_neighbour():
    pages = load_pdf(KB / "unit3_databases.pdf")
    chunks = chunk_documents(pages, contextual_header=False)
    summary = [c for c in chunks if c.metadata["section"] == "Unit summary"]
    assert all(c.page_content.strip() != "storage." for c in chunks)  # the one-word leftover is gone…
    assert summary[-1].page_content.rstrip().endswith("extra\nstorage.")  # …glued onto its neighbour
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))  # still numbered 0..n-1
    # 💡 min_chunk_size=0 switches merging off and brings the tiny chunk back
    unmerged = chunk_documents(pages, contextual_header=False, min_chunk_size=0)
    assert any(c.page_content.strip() == "storage." for c in unmerged)


def test_small_chunks_never_merge_across_sections():
    page = Document(
        page_content="# Notes\n## A\n" + "alpha " * 60 + "\n## B\nTiny section.",
        metadata={"source": "x.pdf", "title": "Notes", "page": 1},
    )
    chunks = chunk_documents([page], chunk_size=200, chunk_overlap=0, contextual_header=False)
    assert chunks[-1].metadata["section"] == "B" and "Tiny section." in chunks[-1].page_content
    assert all("Tiny section." not in c.page_content for c in chunks[:-1])
    assert all(len(c.page_content) >= 200 for c in chunks if c.metadata["section"] == "A")


def test_stemmer_matches_word_families():
    pairs = [
        ("reliable", "reliability"),
        ("process", "processes"),
        ("schedule", "scheduling"),
        ("commit", "committed"),
        ("isolation", "isolated"),
        ("policy", "policies"),
    ]
    for a, b in pairs:
        assert stem(a) == stem(b), (a, b)
    assert tokenize("IPv4, 2NF and TCP's status") == ["ipv4", "2nf", "and", "tcp", "s", "status"]
    assert tokenize("Reliability", stemming=False) == ["reliability"]


def test_bm25_finds_reliability_when_asked_about_reliable(tmp_path):
    store = get_vectorstore("stem_test", embeddings=fake_embeddings(), persist_dir=tmp_path)
    store.add_documents(
        [
            Document("TCP guarantees reliability with acknowledgements.", metadata={"source": "a", "chunk_index": 0}),
            Document("UDP sends datagrams quickly.", metadata={"source": "b", "chunk_index": 0}),
            Document("Routers forward packets between networks.", metadata={"source": "c", "chunk_index": 0}),
        ]
    )
    docs = get_retriever(k=1, mode="bm25", store=store).invoke("Which protocol is reliable?")
    assert docs and docs[0].metadata["source"] == "a"


def test_offline_help_says_bm25_scores_are_real(capsys):
    with pytest.raises(SystemExit):
        evaluation.main(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "BM25 scores are real" in help_text and "mean nothing" not in help_text


# ── Retrieval ──────────────────────────────────────────────────────────


def test_hybrid_retrieval_returns_sources(kb):
    db, _ = kb
    retriever = get_retriever("cs201_test", k=5, mode="hybrid", embeddings=fake_embeddings(), persist_dir=db)
    docs = retriever.invoke("four necessary conditions for deadlock circular wait")
    assert 0 < len(docs) <= 5
    assert all(d.metadata.get("source", "").endswith(".pdf") for d in docs)
    # BM25 alone must find the deadlock section (fake embeddings are random, keywords are not)
    bm25 = get_retriever("cs201_test", k=3, mode="bm25", embeddings=fake_embeddings(), persist_dir=db)
    assert any("Deadlock" in d.metadata["section"] for d in bm25.invoke("circular wait hold and wait"))


def test_metadata_filter(kb):
    db, _ = kb
    retriever = get_retriever("cs201_test", k=5, filter={"unit": 3}, embeddings=fake_embeddings(), persist_dir=db)
    assert {d.metadata["unit"] for d in retriever.invoke("transactions")} == {3}
    assert to_chroma_where({"unit": 1, "source": "x"}) == {"$and": [{"unit": 1}, {"source": "x"}]}


def test_rrf_rewards_agreement():
    a, b, c = (Document(page_content=t, metadata={"source": t, "chunk_index": 0}) for t in "abc")
    fused = reciprocal_rank_fusion([[a, b, c], [b, c, a]])
    assert fused[0].page_content == "b"  # 💡 2nd + 1st beats 1st + 3rd


def test_citation_format():
    doc = Document("x", metadata={"source": "unit1_operating_systems.pdf", "page": 5, "section": "1.6 Deadlocks"})
    assert format_citation(doc) == "unit1_operating_systems.pdf p5 §1.6 Deadlocks"


# ── Metrics ────────────────────────────────────────────────────────────


def _docs(*texts):
    return [Document(page_content=t, metadata={"source": "s.pdf"}) for t in texts]


def test_mrr_on_toy_data():
    docs = _docs("nothing here", "still nothing", "the **Circular**\nwait condition")
    assert evaluation.calculate_mrr("circular wait", docs) == pytest.approx(1 / 3)
    assert evaluation.calculate_mrr("missing", docs) == 0.0


def test_ndcg_on_toy_data():
    # relevant at ranks 2 and 3 → DCG = 1/log2(3) + 1/log2(4); ideal = 1 + 1/log2(3)
    docs = _docs("no", "tcp yes", "tcp again")
    expected = (1 / 1.5849625 + 1 / 2) / (1 + 1 / 1.5849625)
    assert evaluation.calculate_ndcg("tcp", docs, k=3) == pytest.approx(expected, rel=1e-6)
    assert evaluation.calculate_ndcg("tcp", _docs("tcp", "tcp", "no"), k=3) == pytest.approx(1.0)


def test_evaluate_retrieval_and_summary(tmp_path):
    line = {
        "question": "What is TCP?",
        "keywords": ["tcp", "reliable"],
        "reference_answer": "A reliable transport protocol.",
        "source": "s.pdf",
        "category": "direct_fact",
    }
    path = tmp_path / "tests.jsonl"
    path.write_text(json.dumps(line) + "\n\n", encoding="utf-8")
    tests = evaluation.load_tests(path)
    result = evaluation.evaluate_retrieval(tests[0], _docs("udp", "tcp is reliable"), k=5)
    assert result.mrr == pytest.approx(0.5)
    assert result.keyword_coverage == 1.0 and result.source_rank == 1
    summary = evaluation.summarize(tests, [result])
    assert summary["ALL"]["mrr"] == pytest.approx(0.5)


def test_invalid_test_line_fails_loudly(tmp_path):
    path = tmp_path / "tests.jsonl"
    path.write_text('{"question": "Hi?", "category": "nope"}\n', encoding="utf-8")
    with pytest.raises(evaluation._testset.InvalidTestSetError):
        evaluation.load_tests(path)


# ── Agentic RAG and evidence ───────────────────────────────────────────


class ToolFakeChatModel(GenericFakeChatModel):
    """A fake chat model that 'supports' tools: it just replays the scripted messages."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def test_agentic_rag_graph_completes(kb):
    from gradewise.rag import run_agentic_rag

    db, _ = kb
    retriever = get_retriever("cs201_test", k=3, mode="bm25", embeddings=fake_embeddings(), persist_dir=db)
    script = iter(
        [
            _tool_call("retrieve_course_notes", {"query": "deadlock conditions"}, "1"),  # route
            _tool_call("GradeDocuments", {"binary_score": "no"}, "2"),  # grade → rewrite
            AIMessage("Coffman conditions for deadlock"),  # rewrite
            _tool_call("retrieve_course_notes", {"query": "Coffman conditions"}, "3"),  # route again
            _tool_call("GradeDocuments", {"binary_score": "yes"}, "4"),  # grade → answer
            AIMessage("Mutual exclusion, hold and wait, no pre-emption, circular wait [unit1 p5]."),
        ]
    )
    answer, docs, state = run_agentic_rag(
        "What are the deadlock conditions?", llm=ToolFakeChatModel(messages=script), retriever=retriever
    )
    assert answer.startswith("Mutual exclusion")
    assert docs and all(isinstance(d, Document) for d in docs)
    assert state["rewrites"] == 1


def test_agentic_rag_rewrite_cap(kb):
    from gradewise.rag import run_agentic_rag

    db, _ = kb
    retriever = get_retriever("cs201_test", k=2, mode="bm25", embeddings=fake_embeddings(), persist_dir=db)
    script = iter(
        [
            _tool_call("retrieve_course_notes", {"query": "q"}, "1"),
            _tool_call("GradeDocuments", {"binary_score": "no"}, "2"),
            AIMessage("better query"),
            _tool_call("retrieve_course_notes", {"query": "better query"}, "3"),
            # 💡 cap reached: no grading call, straight to the answer
            AIMessage("The course notes don't cover that."),
        ]
    )
    answer, _docs, state = run_agentic_rag(
        "Who won the 1966 World Cup?", llm=ToolFakeChatModel(messages=script), retriever=retriever, max_rewrites=1
    )
    assert "don't cover" in answer and state["rewrites"] == 1


def test_answer_question_with_fake_llm(kb):
    from gradewise.rag import answer_question

    db, _ = kb
    retriever = get_retriever("cs201_test", k=3, mode="bm25", embeddings=fake_embeddings(), persist_dir=db)
    answer, docs = answer_question(
        "What is TCP?", llm=GenericFakeChatModel(messages=iter(["TCP is reliable."])), retriever=retriever
    )
    assert answer == "TCP is reliable." and docs


def test_find_evidence_uses_question_not_just_answer(kb):
    from gradewise.rag import find_evidence

    db, _ = kb
    retriever = get_retriever("cs201_test", k=5, mode="bm25", embeddings=fake_embeddings(), persist_dir=db)
    question = Question(
        id="Q4",
        number="4",
        text="State the four necessary conditions for deadlock.",
        max_marks=4,
        question_type="short",
        criteria=[
            MarkingCriterion(
                id="Q4.c1", description="Names the conditions", marks=4, keywords=["circular wait", "hold and wait"]
            )
        ],
        methodology="1 mark each.",
        reference_topics=["deadlock"],
    )
    # A wrong, off-topic answer must not drag the evidence away from deadlocks.
    evidence = find_evidence(question, "SQL joins combine tables using foreign keys.", k=3, retriever=retriever)
    assert len(evidence) == 3
    assert evidence[0].metadata["source"] == "unit1_operating_systems.pdf"
    assert all("citation" in d.metadata and d.metadata["matched_queries"] is not None for d in evidence)
