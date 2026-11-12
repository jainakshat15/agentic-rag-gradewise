"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🖥️ Week 6 · Day 2: The GradeWise dashboard                          │
│                                                                      │
│  What this file does                                                 │
│    The shell around every page: page config, the theme's CSS, the    │
│    sidebar (API status, 🧪 demo toggle, MODEL_MODE + DATA_SIZE       │
│    chips), and navigation.                                           │
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
st.session_state["health"] = health  # 💡 pages read MODEL_MODE / DATA_SIZE from here


def mode_chip(h: dict) -> str:
    """'cheap · gpt-6-luna' / 'free · openrouter/free' / 'custom · <chat model>' (from GET /health)."""
    mode = h.get("model_mode", "cheap")
    chat = str(h.get("chat_model", "")).split(":", 1)[-1]
    short = chat.rsplit("/", 1)[-1]  # 'openai/gpt-6-luna' → 'gpt-6-luna'
    label = {"cheap": f"cheap · {short}", "free": "free · openrouter/free"}.get(mode, f"custom · {chat}")
    return styles.pill(label, {"cheap": "violet", "free": "cyan"}.get(mode, "muted"))


# 💡 Demo mode starts as the server's default (GRADEWISE_DEMO) or, with no API key, ON.
# ⚠️ The truth lives in a plain key ("demo_mode"), not in the toggle's own key: Streamlit may drop a widget's
#    state when you switch pages, which silently turned demo mode OFF (and broke every no-key call).
if "demo_mode" not in st.session_state:
    st.session_state["demo_mode"] = bool(health and (health["demo"] or not health["provider_key"])) or health is None


def _sync_demo() -> None:
    st.session_state["demo_mode"] = st.session_state["demo_toggle"]


with st.sidebar:
    st.html('<div class="gw-brand">🎓 Grade<span>Wise</span></div>')
    st.caption("Agentic RAG grading for Emma's CS201 class at the Northbridge Institute.")
    st.toggle(
        "🧪 Demo mode",
        value=st.session_state["demo_mode"],
        key="demo_toggle",
        on_change=_sync_demo,
        help="Offline fakes instead of real models: free, no API key, and the marks mean nothing. "
        "Demo data lives in output/demo/, apart from real grading.",
    )
    if health:
        mode = styles.pill("demo", "amber") if st.session_state["demo_mode"] else styles.pill("live models", "green")
        chips = mode_chip(health)
        if health.get("data_size") == "small":
            chips += styles.pill("small data", "amber")
        st.html(f"{styles.pill('API online', 'green')}{mode}<br>{chips}")
        if health.get("data_size") == "small" and health.get("data_banner"):
            st.caption(health["data_banner"] + " · switch with DATA_SIZE=full in .env")
        if health.get("model_mode") == "free" and health.get("data_size") != "small":
            st.caption("⚠️ Free models allow about 50 requests a day: set DATA_SIZE=small in .env.")
        if not st.session_state["demo_mode"]:
            st.caption(f"Grading with `{health['grading_model']}` · vision `{health['vision_model']}`")
            if not health["provider_key"]:
                needed = " / ".join(health.get("required_keys") or []) or "an API key"
                st.warning(
                    f"MODEL_MODE={health.get('model_mode', 'cheap')} needs {needed} in .env: add it, "
                    "or turn demo mode on.",
                    icon="🔑",
                )
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
