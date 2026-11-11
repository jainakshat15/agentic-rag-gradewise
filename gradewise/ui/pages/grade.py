"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  ✍️ Week 6 · Day 3: Grade a batch, with live progress                │
│                                                                      │
│  What: start a batch (upload answer PDFs, or use the synthetic       │
│        class), then watch it LIVE: a progress bar per student and an │
│        event log, streamed from the API over Server-Sent Events.     │
│  ⏮ Built on: POST /batches, GET /batches/{id}/events (SSE)          │
│  ⏭ Next: 🔍 Review queue, when the batch pauses for Asha            │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import streamlit as st

from gradewise.ui import client, styles
from gradewise.ui.common import demo, exam_picker, friendly_errors, live_progress, status_pill

styles.hero(
    "Grade a batch",
    "Transcribe → grade every question in parallel → pause for anything uncertain → write report cards.",
    kicker="Weeks 4 & 5 · The orchestrator",
)

SOURCES = {
    "transcripts": "📄 Synthetic class, from transcripts (skip the vision model: cheapest)",
    "synthetic": "🖼️ Synthetic class, from the handwritten PDFs (uses the vision model)",
    "upload": "⬆️ Upload answer-sheet PDFs (student ID = file name)",
}

with st.form("batch"):
    exam_id = exam_picker()
    source = st.radio("Answer sheets", list(SOURCES), format_func=SOURCES.get)
    c1, c2 = st.columns(2)
    students = c1.slider("How many synthetic students", 1, 10, 3)
    auto = c2.toggle("Auto-approve every review", help="Skips the human. Fine for a demo, never for real marks.")
    uploads = st.file_uploader("Answer-sheet PDFs", type=["pdf"], accept_multiple_files=True)
    if demo():
        st.caption("🧪 Demo mode: a keyword 'grader' marks the answers and is unsure about low marks, so expect flags.")
    start = st.form_submit_button("🚀 Start grading", type="primary", disabled=exam_id is None)

if start:
    data = {
        "exam_id": exam_id,
        "students": students,
        "auto_approve": auto,
        "from_transcripts": source == "transcripts",
        "use_synthetic": source == "synthetic",
        "demo": demo(),
    }
    files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploads or []] if source == "upload" else []
    with friendly_errors():
        created = client.post("/batches", data=data, files=files)
    st.session_state["batch_id"] = created["batch_id"]

batch_id = st.session_state.get("batch_id")
if batch_id:
    st.divider()
    with friendly_errors():
        status = client.get(f"/batches/{batch_id}")
    st.html(f"<h4>Batch <code>{batch_id}</code> {status_pill(status['status'])}</h4>")
    # 💡 Follows the stream while the batch runs; on a finished batch it ends at once and just redraws.
    with friendly_errors():
        final = live_progress(batch_id)
        status = client.get(f"/batches/{batch_id}")

    if final == "paused":
        pending = status.get("pending") or {}
        st.warning(
            f"⏸️ Paused: **{pending.get('student_id')}** needs a "
            f"{pending.get('type', 'review').replace('_', ' ')} ({len(pending.get('items', []))} item(s)).",
        )
        st.page_link("pages/review.py", label="Open the review queue →", icon="🔍")
    elif final == "done":
        st.success(f"✅ Done: {len(status['reports'])} report card(s) stored.")
        styles.table(
            ["Student", "Total", "%", "Band"],
            [
                (r["student_id"], f"{r['total']:g}/{r['max_total']:g}", f"{r['percentage']:g}", r["grade_band"])
                for r in status["reports"]
            ],
        )
        st.page_link("pages/reports.py", label="See the reports →", icon="📊")
    elif final == "failed" and status.get("error"):
        from gradewise.ui.client import ApiError
        from gradewise.ui.common import show_error

        show_error(ApiError(status["error"]))
