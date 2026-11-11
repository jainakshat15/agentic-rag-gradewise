"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🖥️ Week 6 · Day 2: gradewise.ui, Asha's dashboard (Streamlit)       │
│                                                                      │
│  What this package does                                              │
│      app.py      → the shell: theme, sidebar, demo toggle, pages     │
│      pages/      → one file per page (Home … Chat with results)      │
│      client.py   → the ONLY way the UI reaches GradeWise: HTTP       │
│      common.py   → shared widgets (errors, exam picker, live log)    │
│      styles.py   → CSS + tiny HTML components                        │
│                                                                      │
│  Run : uv run python -m gradewise.api --demo         (terminal 1)    │
│        uv run python -m streamlit run gradewise/ui/app.py  (term. 2) │
│  ⏮ Built on: gradewise/api   ⏭ Used by: docker-compose.yml `ui`     │
╰──────────────────────────────────────────────────────────────────────╯
"""
