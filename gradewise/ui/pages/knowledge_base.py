"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📚 Week 6 · Day 2: Knowledge bases, upload and ingest               │
│                                                                      │
│  What: upload course-note PDFs into a named collection and ingest    │
│        them (week 2's idempotent pipeline: unchanged files skipped). │
│  ⏮ Built on: GET /knowledge-bases, POST /knowledge-bases/{name}/...  │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import streamlit as st

from gradewise.ui import client, styles
from gradewise.ui.common import demo, friendly_errors

styles.hero(
    "Knowledge bases",
    "The course notes every mark is checked against. The grader retrieves from here and cites what it finds.",
    kicker="Week 2 · Ingestion",
)

with friendly_errors():
    kbs = client.get("/knowledge-bases", demo=demo())

if kbs:
    styles.table(
        ["Collection", "Chunks", "Files"],
        [(kb["name"], kb["chunks"], ", ".join(kb["files"]) or "(ingested outside the API)") for kb in kbs],
    )
else:
    st.info("No knowledge base yet. The exam's default collection is **cs201**: start there.", icon="📚")

st.markdown("#### ➕ Add documents")
with st.form("ingest"):
    c1, c2 = st.columns([2, 3])
    name = c1.text_input("Collection name", value="cs201", help="Letters, digits, '-' and '_'.")
    files = c2.file_uploader("Course-note PDFs", type=["pdf"], accept_multiple_files=True)
    notes = st.checkbox("Also add the four CS201 course-note PDFs that ship with the course", value=not kbs)
    reset = st.checkbox("Reset the collection first (re-embed everything)")
    if demo():
        st.caption("🧪 Demo mode: fake embeddings (free). Keyword search works; semantic search is noise.")
    else:
        st.caption("💰 Real embeddings: a few cents for the CS201 notes. Unchanged files are never re-embedded.")
    submitted = st.form_submit_button("📥 Upload and ingest", type="primary")

if submitted:
    upload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files or []]
    with st.spinner("Parsing, chunking and embedding… (the first run takes a minute)"), friendly_errors():
        result = client.post(
            f"/knowledge-bases/{name}/documents",
            data={"include_course_notes": notes, "reset": reset, "demo": demo()},
            files=upload,
        )
    st.success(f"**{result['name']}** now holds **{result['chunks']}** chunks.", icon="✅")
    styles.stat_cards(
        [
            ("Added", len(result["added"]), ", ".join(result["added"]) or "–"),
            ("Updated", len(result["updated"]), ", ".join(result["updated"]) or "–"),
            ("Skipped (unchanged)", len(result["skipped"]), "no cost"),
            ("New chunks", result["chunks_added"], f"≈ ${result['est_cost_usd']:.4f}"),
        ]
    )
