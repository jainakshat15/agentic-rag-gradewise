"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🏠 Week 6 · Day 2: Home, the pipeline at a glance                   │
│                                                                      │
│  What: the six stages of GradeWise as a diagram, each lit up by what │
│        the API reports (ingested? approved? graded? waiting?).       │
│  ⏮ Built on: GET /health, /knowledge-bases, /exams, /batches        │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import streamlit as st

from gradewise.ui import client, styles
from gradewise.ui.common import demo, friendly_errors, status_pill

styles.hero(
    "From a pile of scripts to report cards",
    "GradeWise reads Emma's handwritten exam answers, grades them against the course notes with evidence, "
    "and asks her before anything uncertain becomes a mark.",
    kicker="Week 6 · Ship it",
)

with friendly_errors():
    kbs = client.get("/knowledge-bases", demo=demo())
    exams = client.get("/exams", demo=demo())
    batches = [b for b in client.get("/batches") if b["demo"] == demo()]

chunks = sum(kb["chunks"] for kb in kbs)
approved = [e for e in exams if e["source"] == "database"]
paused = [b for b in batches if b["status"] == "paused"]
running = [b for b in batches if b["status"] in ("queued", "running")]
graded = sum(len(b["reports"]) for b in batches)

styles.flow(
    [
        ("📚", "Knowledge base", f"{chunks} chunks" if chunks else "not ingested", "done" if chunks else ""),
        ("📝", "Exam", f"{len(approved)} approved" if approved else "sample only", "done" if approved else ""),
        ("✍️", "Transcribe", "vision LLM · verbatim", "done" if batches else ""),
        ("🧮", "Grade", "rubric + evidence", "wait" if running else ("done" if graded else "")),
        ("🧑‍🏫", "Review", f"{len(paused)} waiting" if paused else "human in the loop", "wait" if paused else ""),
        ("📄", "Report cards", f"{graded} this session" if graded else "HTML + PDF", "done" if graded else ""),
        ("💬", "Chat", "ask about results", "done" if graded else ""),
    ]
)

styles.stat_cards(
    [
        ("Knowledge bases", len(kbs), ", ".join(kb["name"] for kb in kbs) or "none yet"),
        ("Exams", len(exams), f"{len(approved)} approved here"),
        ("Batches", len(batches), f"{len(running)} running · {len(paused)} paused"),
        ("Report cards", graded, "in this API session"),
    ]
)

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown("#### 🚀 Your first run, in four clicks")
    st.markdown(
        "1. **📚 Knowledge base**: ingest the CS201 course notes (the evidence the grader cites).\n"
        "2. **📝 Exams**: load the sample midterm pack, check the extracted questions, approve.\n"
        "3. **✍️ Grade a batch**: grade a few synthetic students and watch the progress stream in.\n"
        "4. **🔍 Review queue** → **📊 Reports** → **💬 Chat**: decide on the flags, read the cards, ask questions."
    )
    if demo():
        st.info(
            "**Demo mode is on.** Every model is an offline fake: free and instant, but the marks are keyword "
            "matches, not judgement. Turn it off in the sidebar once your API key is in `.env`.",
            icon="🧪",
        )
with right:
    st.markdown("#### 🕒 Recent batches")
    if not batches:
        st.caption("Nothing graded yet in this mode.")
    for b in batches[:6]:
        st.html(
            f'<div class="gw-card" style="margin-bottom:8px"><b>{b["batch_id"]}</b> {status_pill(b["status"])}'
            f'<div class="gw-muted">{b["exam_id"]} · {len(b["students"])} student(s)</div></div>'
        )
