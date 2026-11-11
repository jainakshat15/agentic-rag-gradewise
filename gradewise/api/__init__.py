"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🚢 Week 6 · Day 1: gradewise.api, the FastAPI backend               │
│                                                                      │
│  What this package does                                              │
│    Puts a web API in front of everything weeks 1–5 built, so a       │
│    dashboard (or any other client) can drive GradeWise over HTTP:    │
│                                                                      │
│      main.py       → the app: create_app(), every route              │
│      schemas.py    → Pydantic request / response models              │
│      jobs.py       → background batches + their progress events      │
│      workspace.py  → where files go (real vs demo), demo-mode switch │
│      demo.py       → offline fakes, so it all runs with no API key   │
│                                                                      │
│  Run : uv run python -m gradewise.api        (http://localhost:8000) │
│        then open http://localhost:8000/docs for the interactive docs │
│  ⏮ Built on: gradewise/orchestrator.py, exam/, reports/, ingestion/ │
│  ⏭ Used by: gradewise/ui (Streamlit), docker-compose.yml            │
╰──────────────────────────────────────────────────────────────────────╯
"""

# 💡 Nothing is imported here on purpose: `import gradewise.api` stays instant, and the app
#    (which pulls in LangGraph, Chroma, ...) is only built when you ask for gradewise.api.main.
