"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧪 tests/test_modes.py: MODEL_MODE, DATA_SIZE and store keying      │
│                                                                      │
│  Run:  uv run pytest tests/test_modes.py                             │
│  💡 Offline and free: no key, no network.                            │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from gradewise import datasets, pricing
from gradewise.config import MODE_MODELS, Settings, settings
from gradewise.ingestion import embedding_key, get_vectorstore, ingest, store_dir
from gradewise.schemas import ExamPaper


# ── 🧠 MODEL_MODE ──────────────────────────────────────────────────────────
def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_cheap_is_the_default_and_uses_gpt_6_luna_via_openrouter(monkeypatch):
    for var in ("MODEL_MODE", "CHAT_MODEL", "SMART_MODEL", "VISION_MODEL", "EMBEDDING_MODEL"):
        monkeypatch.delenv(var, raising=False)
    s = _settings()
    assert s.model_mode == "cheap"
    assert s.chat_model == s.grading_model == s.transcription_model == "openrouter:openai/gpt-6-luna"
    assert s.embedding_model == "openrouter:openai/text-embedding-3-small"
    assert s.required_keys == ["OPENROUTER_API_KEY"]


def test_free_mode_uses_the_free_router_and_the_same_key():
    s = _settings(model_mode="free")
    assert s.chat_model == s.smart_model == s.vision_model == "openrouter:openrouter/free"
    assert s.embedding_model == "openrouter:nvidia/nemotron-3.5-embed-1b:free"
    assert s.required_keys == ["OPENROUTER_API_KEY"]
    assert s.max_concurrency(4) == 2 and s.max_concurrency(1) == 1


@pytest.mark.parametrize("mode", ["cheap", "free"])
def test_preset_modes_ignore_the_model_variables(mode):
    s = _settings(model_mode=mode, chat_model="anthropic:x", smart_model="anthropic:y", embedding_model="ollama:z")
    assert s.models == MODE_MODELS[mode]
    assert s.max_concurrency(4) == (2 if mode == "free" else 4)


def test_custom_mode_uses_the_model_variables(monkeypatch):
    monkeypatch.setenv("MODEL_MODE", "custom")
    monkeypatch.setenv("SMART_MODEL", "openrouter:openai/gpt-6.1-terra")
    monkeypatch.setenv("CHAT_MODEL", "ollama:llama3.2")
    monkeypatch.setenv("VISION_MODEL", "google_genai:gemini-3.1-flash-lite")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    s = _settings()
    assert s.smart_model == s.grading_model == "openrouter:openai/gpt-6.1-terra"
    assert s.chat_model == "ollama:llama3.2"
    assert s.embedding_model == "openrouter:openai/text-embedding-3-small"  # unset → the default
    assert s.required_keys == ["OPENROUTER_API_KEY", "GOOGLE_API_KEY"]  # ollama needs none


def test_old_budget_mode_is_accepted_and_ignored(monkeypatch):
    monkeypatch.setenv("BUDGET_MODE", "true")
    s = _settings(budget_mode=True, model_mode=" Free ")
    assert not hasattr(s, "budget_mode")
    assert s.model_mode == "free"


def test_empty_values_mean_the_defaults():
    s = _settings(model_mode="", data_size="")
    assert (s.model_mode, s.data_size) == ("cheap", "full")


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="cheap"):
        _settings(model_mode="pricey")


def test_banner_names_mode_and_models():
    assert "MODEL_MODE=cheap" in _settings().describe_models()
    assert "gpt-6-luna" in _settings().describe_models()
    assert "openrouter/free" in _settings(model_mode="free").describe_models()


@pytest.mark.parametrize("mode", ["cheap", "free"])
def test_missing_key_error_names_the_openrouter_key(monkeypatch, mode):
    from gradewise.errors import MissingAPIKeyError
    from gradewise.models import chat_model

    monkeypatch.setattr(settings, "model_mode", mode)
    monkeypatch.setattr(settings, "openrouter_api_key", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError) as exc:
        chat_model()
    text = str(exc.value)
    assert f"MODEL_MODE={mode}" in text and "OPENROUTER_API_KEY" in text and "openrouter.ai/keys" in text


# ── 💰 Prices ──────────────────────────────────────────────────────────────
def test_prices_follow_the_spec():
    luna = "openrouter:openai/gpt-6-luna"
    assert (pricing.chat_price(luna).input, pricing.chat_price(luna).output) == (0.10, 0.50)
    assert pricing.chat_price("openrouter:openai/gpt-6.1-terra").output == 5.0
    assert pricing.chat_price("openai:gpt-6-luna").input == 0.10  # same model, called directly
    assert pricing.embedding_price("openrouter:openai/text-embedding-3-small") == 0.02
    assert pricing.chat_cost(1_000_000, 1_000_000, luna) == pytest.approx(0.60)
    assert pricing.chat_cost(1_000_000, 0, luna, cached_tokens=1_000_000) == pytest.approx(0.01)
    assert pricing.chat_cost(10**6, 10**6, "openrouter:openrouter/free") == 0.0
    assert pricing.embedding_price("openrouter:nvidia/nemotron-3.5-embed-1b:free") == 0.0
    assert pricing.chat_price("google_genai:gemini-3.1-flash-lite").input == 0.25  # custom, paid tier


def test_free_mode_prices_are_zero(monkeypatch):
    monkeypatch.setattr(settings, "model_mode", "free")
    assert pricing.chat_cost(10**6, 10**6) == 0.0
    assert pricing.embedding_price() == 0.0
    assert pricing.format_cost(0.0, settings.embedding_model) == "$0 (free tier)"


# ── 📦 DATA_SIZE ───────────────────────────────────────────────────────────
@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(settings, "data_size", "small")


def test_full_data_is_everything():
    assert not datasets.is_small()
    assert [p.name for p in datasets.kb_files()] == [
        "cs201_course_handbook.pdf",
        "unit1_operating_systems.pdf",
        "unit2_computer_networks.pdf",
        "unit3_databases.pdf",
    ]
    assert len(datasets.exam_question_ids()) == 15
    assert datasets.synthetic_students() == [f"student_{i:02d}" for i in range(1, 11)]
    assert datasets.eval_slice(range(9)) == list(range(9))
    assert datasets.describe() == "📦 DATA_SIZE=full: 4 PDFs · all 10 questions (15 parts) · 10 students"


def test_small_data_is_the_spec_subset(small):
    assert datasets.is_small()
    assert [p.name for p in datasets.kb_files()] == ["cs201_course_handbook.pdf", "unit1_operating_systems.pdf"]
    assert datasets.exam_question_ids() == ["Q1", "Q2a", "Q2b", "Q3"]
    assert datasets.synthetic_students() == ["student_01", "student_05"]
    assert [p.name for p in datasets.synthetic_sheets()] == ["student_01.pdf", "student_05.pdf"]
    assert datasets.eval_slice(range(9)) == [0, 1, 2, 3, 4]
    assert datasets.describe() == "📦 DATA_SIZE=small: 2 PDFs · Q1–Q3 · 2 students"


def test_explicit_small_argument_overrides_the_env(small):
    assert len(datasets.kb_files(small=False)) == 4
    assert datasets.exam_question_ids(small=False)[-1] == "Q10b"
    assert datasets.is_small(False) is False


def test_own_folder_is_never_cut_down(small, tmp_path):
    (tmp_path / "my_notes.pdf").write_bytes(b"%PDF-1.4")
    assert [p.name for p in datasets.kb_files(tmp_path)] == ["my_notes.pdf"]


def test_subset_exam_keeps_all_parts_and_fixes_totals(small):
    exam = ExamPaper.model_validate_json(datasets.SAMPLE_EXAM.read_text(encoding="utf-8"))
    sub = datasets.subset_exam(exam)
    assert [q.id for q in sub.questions] == ["Q1", "Q2a", "Q2b", "Q3"]
    assert sub.total_marks == sub.questions_total and not sub.problems()
    assert datasets.subset_exam(exam, ["Q7a"]).total_marks == exam.questions[9].max_marks
    with pytest.raises(ValueError):
        datasets.subset_exam(exam, ["Q99"])


def test_golden_rows_small_is_2_students_by_q1_to_q3(small):
    from gradewise.evals.grading_eval import load_golden

    rows = datasets.golden_rows(load_golden())
    assert set(rows["student_id"]) == {"student_01", "student_05"}
    assert set(rows["question_id"]) == {"Q1", "Q2a", "Q2b", "Q3"}
    assert len(datasets.golden_rows(load_golden(), small=False)) > len(rows)


def test_sheets_from_transcripts_follow_data_size(small):
    from gradewise.orchestrator import sheets_from_transcripts

    exam = ExamPaper.model_validate_json(datasets.SAMPLE_EXAM.read_text(encoding="utf-8"))
    assert [s.student_id for s in sheets_from_transcripts(exam)] == ["student_01", "student_05"]
    assert len(sheets_from_transcripts(exam, students=3)) == 3  # 💡 an explicit argument wins


def test_grade_batch_grades_q1_to_q3_when_small(small, tmp_path):
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    from gradewise.grading.fakes import KeywordGraderModel
    from gradewise.orchestrator import build_orchestrator, grade_batch, sheets_from_transcripts

    class NoEvidence(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager=None):
            return [Document("Deadlock needs four conditions.", metadata={"source": "unit1.pdf", "page": 1})]

    exam = ExamPaper.model_validate_json(datasets.SAMPLE_EXAM.read_text(encoding="utf-8"))
    app = build_orchestrator(
        "memory",
        llm=KeywordGraderModel(),
        retriever=NoEvidence(),
        feedback="rules",
        db=tmp_path / "gw.db",
        embeddings=DeterministicFakeEmbedding(size=16),
        persist_dir=tmp_path / "vdb",
        reports_dir=tmp_path / "reports",
        pdf=False,
    )
    sheets = sheets_from_transcripts(exam)
    result = grade_batch(exam.exam_id, sheets, thread_id="small", exam=exam, orchestrator=app, auto_approve=True)
    assert result.status == "done"
    assert {r.question_id for r in result.reports[0].results} == {"Q1", "Q2a", "Q2b", "Q3"}
    # 💡 explicit questions win over DATA_SIZE
    result = grade_batch(
        exam.exam_id, sheets[:1], thread_id="one", exam=exam, orchestrator=app, auto_approve=True, questions=["Q5"]
    )
    assert [r.question_id for r in result.reports[0].results] == ["Q5"]


def test_ingest_small_embeds_two_pdfs_and_keeps_the_rest(monkeypatch, tmp_path):
    emb = DeterministicFakeEmbedding(size=16)
    full = ingest(
        collection="kb_small", embeddings=emb, persist_dir=tmp_path, quiet=True, files=["unit1_operating_systems.pdf"]
    )
    assert full.added == ["unit1_operating_systems.pdf"]
    monkeypatch.setattr(settings, "data_size", "small")
    report = ingest(collection="kb_small", embeddings=emb, persist_dir=tmp_path, quiet=True)
    assert report.files == 2 and report.added == ["cs201_course_handbook.pdf"]
    assert report.skipped == ["unit1_operating_systems.pdf"] and report.removed == []


# ── 🔢 Vector stores keyed by embedding model ──────────────────────────────
def test_embedding_keys():
    assert embedding_key(model="openai:text-embedding-3-small") == "openai_text-embedding-3-small"
    assert embedding_key(model="openrouter:openai/text-embedding-3-small") == "openrouter_openai_text-embedding-3-small"
    assert embedding_key(model="google_genai:models/gemini-embedding-001") == "google_genai_gemini-embedding-001"
    assert embedding_key(DeterministicFakeEmbedding(size=32)) == "DeterministicFakeEmbedding_32"
    assert embedding_key() == "openrouter_openai_text-embedding-3-small"  # cheap mode's model


def test_embedding_key_matches_for_objects_and_strings():
    from langchain_openai import OpenAIEmbeddings

    from gradewise.models import RetryingEmbeddings

    emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key="sk-offline")
    assert embedding_key(emb) == embedding_key(model="openai:text-embedding-3-small")
    assert embedding_key(RetryingEmbeddings(emb)) == embedding_key(emb)


def test_switching_mode_switches_store_folder(monkeypatch, tmp_path):
    cheap = store_dir(tmp_path)
    monkeypatch.setattr(settings, "model_mode", "free")
    free = store_dir(tmp_path)
    assert cheap != free and free.name == "openrouter_nvidia_nemotron-3.5-embed-1b_free"


def test_different_embedding_sizes_never_share_a_collection(tmp_path):
    small_store = get_vectorstore("cs201", embeddings=DeterministicFakeEmbedding(size=8), persist_dir=tmp_path)
    small_store.add_texts(["deadlock needs four conditions"])
    big_store = get_vectorstore("cs201", embeddings=DeterministicFakeEmbedding(size=24), persist_dir=tmp_path)
    big_store.add_texts(["a different model, a different space"])  # 💥 would raise a dimension error if shared
    assert len(small_store.get()["ids"]) == 1 and len(big_store.get()["ids"]) == 1
    assert {p.name for p in tmp_path.iterdir()} == {"DeterministicFakeEmbedding_8", "DeterministicFakeEmbedding_24"}
