# 🆕 Changelog

Every update to the course repo, newest first. If you're mid-course, `git pull` then `uv sync` picks everything up.

## 2026-11-11: The full course, weeks 1–6
- **Week 1 · Foundations:** Days 1–5 notebooks, `week1_EXERCISE` (multiple-choice questions in the schema), solutions, and `week1/project/extract_exam.py` (exam pack → `ExamPaper` JSON). The CS201 exam pack lives in `data/exams/cs_midterm/` (built by `scripts/build_exam_pack.py`), with the hand-checked `expected_exam.json`
- **Week 2 · RAG:** Days 1–5, `week2_EXERCISE` (beat the baseline nDCG), solutions; `implementation/` (simple) and `pro_implementation/` (pro) ingest + answer scripts; `evaluation/` with 60 test questions and `eval.py` (MRR, nDCG, Source@k); `gradewise/ingestion/` and `gradewise/rag/`; the `ingest` CLI command
- **Week 3 · Agents & LangGraph:** Days 1–5, `week3_EXERCISE` (edit a question during the interrupt), solutions; the exam graph (extract → validate → retry → admin approval → store) in `gradewise/exam/` and `week3/project/exam_graph.py`; the `exam extract` CLI command
- **Week 4 · Agentic RAG & handwriting:** Days 1–5, `week4_EXERCISE` (compare two vision models on CER and cost), solutions; the agentic RAG graph and evidence retriever (`gradewise/rag/`), the transcription subgraph (`gradewise/transcription/`); ten synthetic answer sheets with ground truth (`scripts/make_synthetic_sheets.py`); the `transcribe` CLI command
- **Week 5 · Grading & evaluation:** Days 1–5, `week5_EXERCISE` (a moderation second-grader), solutions; the grading agent and guardrails (`gradewise/grading/`), report cards and the "chat with results" agent (`gradewise/reports/`), the orchestrator with human review (`gradewise/orchestrator.py`), MAE / QWK / CER metrics (`gradewise/evals/`) and the golden set `data/golden/synthetic_labels.csv`; the `grade`, `report`, `ask` and `evals` CLI commands
- **Week 6 · Ship it:** Days 1–5, `week6_EXERCISE` (ship your own variant), solutions incl. a complete BIO101 variant; the FastAPI backend (`gradewise/api/`) with an offline 🧪 demo mode, the Streamlit dashboard with a review queue (`gradewise/ui/`), `docker/Dockerfile` + `docker-compose.yml`, `week6/project/DEPLOY.md`; the `serve` CLI command
- **Quizzes:** `quizzes/week_1.yaml` … `week_6.yaml` and `final.yaml` (76 questions, an explanation for every option)
- **Guides:** 14 self-study notebooks in `guides/`, from the command line to Docker
- **Diagrams:** every Mermaid diagram is shipped as a PNG in `assets/diagrams/` (with its source in a collapsible block), because Cursor / VS Code show Mermaid as raw text; `gradewise.display.show_graph()` draws LangGraph graphs inline, offline with `GRADEWISE_OFFLINE_DIAGRAMS=1` (`scripts/render_diagrams.py`)
- **Assets:** day banners, weekly roadmaps, callout icons and dashboard screenshots (`scripts/make_assets.py`)
- **Tests:** 169 free, offline tests (`uv run python -m pytest -q`)

## 2026-11-11: CS201 knowledge base
- Original CS201 course notes, written for this course: Unit 1 Operating Systems, Unit 2 Computer Networks, Unit 3 Database Systems, plus the course handbook with the general marking principles (≈7,100 words, 22 pages)
- Rendered to styled PDFs in `data/knowledge-base/` by `scripts/build_knowledge_base.py`

## 2026-11-11: Repo foundations
- Repo skeleton: root README, `AGENTS.md`, contributing guide
- `setup/`: Windows, Mac and Linux guides, provider-swap guide, `diagnostics.py`, troubleshooting notebook
- `gradewise/` core: settings (`config.py`), console voice (`console.py`), friendly errors (`errors.py`), model factory (`models.py`), CLI (`python -m gradewise`)
- Week 1, Day 1 notebook
- Pinned `pymupdf` to the 1.25 series: newer wheels are blocked by Windows Smart App Control on some machines (see `setup/troubleshooting.ipynb` §4)
- Tested with: `langchain` 1.4, `langgraph` 1.2, Python 3.12
