"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🖥️ Week 6 · Day 2: The GradeWise dashboard                          │
│                                                                      │
│  What this file does                                                 │
│    The shell around every page: page config, the theme's CSS, the    │
│    sidebar (API status + the 🧪 demo-mode toggle), and navigation.   │
│    Streamlit re-runs this file top to bottom on EVERY click, then    │
│    runs the page you're on.                                          │
│                                                                      │
│  Run : uv run python -m streamlit run gradewise/ui/app.py            │
│        (start the API first: uv run python -m gradewise.api --demo)  │
│  Env : API_URL (default http://localhost:8000)                       │
│  ⏮ Built on: gradewise/api   ⏭ Next: docker compose up (day 4)      │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from gradewise.ui import client, styles
from gradewise.ui.client import ApiError

PAGES = Path(__file__).parent / "pages"

st.set_page_config(page_title="GradeWise", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
styles.inject()

# ── 🔌 Is the API there? ─────────────────────────────────────────────────────────
try:
    health = client.get("/health")
except ApiError as err:
    health = None
    health_error = err

# 💡 Demo mode starts as the server's default (GRADEWISE_DEMO) or, with no API key, ON.
if "demo" not in st.session_state:
    st.session_state["demo"] = bool(health and (health["demo"] or not health["provider_key"])) or health is None

with st.sidebar:
    st.html('<div class="gw-brand">🎓 Grade<span>Wise</span></div>')
    st.caption("Agentic RAG grading for Asha's CS201 class at the Northbridge Institute.")
    st.toggle(
        "🧪 Demo mode",
        key="demo",
        help="Offline fakes instead of real models: free, no API key, and the marks mean nothing. "
        "Demo data lives in output/demo/, apart from real grading.",
    )
    if health:
        mode = styles.pill("demo", "amber") if st.session_state["demo"] else styles.pill("live models", "green")
        st.html(f"{styles.pill('API online', 'green')}{mode}")
        if not st.session_state["demo"]:
            st.caption(f"Grading with `{health['grading_model']}` · vision `{health['vision_model']}`")
            if not health["provider_key"]:
                st.warning("No API key is configured: turn demo mode on, or add a key to .env.", icon="🔑")
        running = health["jobs"].get("running", 0)
        paused = health["jobs"].get("paused", 0)
        if running or paused:
            st.caption(f"⏳ {running} batch(es) running · ⏸️ {paused} waiting for review")
    else:
        st.html(styles.pill("API offline", "red"))
        st.caption(f"`{client.API_URL}`")

pages = [
    st.Page(PAGES / "home.py", title="Home", icon="🏠", default=True),
    st.Page(PAGES / "knowledge_base.py", title="Knowledge base", icon="📚"),
    st.Page(PAGES / "exams.py", title="Exams", icon="📝"),
    st.Page(PAGES / "grade.py", title="Grade a batch", icon="✍️"),
    st.Page(PAGES / "review.py", title="Review queue", icon="🔍"),
    st.Page(PAGES / "reports.py", title="Reports", icon="📊"),
    st.Page(PAGES / "chat.py", title="Chat with results", icon="💬"),
]
page = st.navigation(pages)

if health is None:
    styles.hero("The API isn't answering", "The dashboard is only a window: GradeWise itself runs in the API.")
    from gradewise.ui.common import show_error

    show_error(health_error)
    st.code("uv run python -m gradewise.api --demo", language="bash")
    st.stop()

page.run()
