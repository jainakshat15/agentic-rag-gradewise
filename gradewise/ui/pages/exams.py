"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📝 Week 6 · Day 2: Exams, upload → check → approve                  │
│                                                                      │
│  What: upload the exam pack (question paper, marking scheme,         │
│        methodology, optional model answers). The week 3 graph        │
│        extracts it and PAUSES; Emma checks the questions and         │
│        approves. Only approved exams are stored and graded against.  │
│  ⏮ Built on: POST /exams, POST /exams/{id}/approve, GET /exams      │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from html import escape

import streamlit as st

from gradewise.ui import client, styles
from gradewise.ui.common import demo, friendly_errors

styles.hero(
    "Exams",
    "Turn three PDFs into a structured exam: every question, its marks, and the criteria that earn them.",
    kicker="Weeks 1 & 3 · Exam understanding",
)


def show_questions(summary: dict) -> None:
    rows = []
    for q in summary["questions"]:
        ok = q["criteria_total"] == q["max_marks"]
        check = styles.pill("✓ adds up", "green") if ok else styles.pill(f"criteria = {q['criteria_total']:g}", "red")
        rows.append(
            (
                f"<b>{q['id']}</b>",
                styles.pill(str(q["type"]).replace("_", " "), "cyan"),
                f"{q['max_marks']:g}",
                f"{q['criteria']} {check}",
                escape(q["text"]),
            )
        )
    styles.table(["ID", "Type", "Marks", "Criteria", "Question"], rows, raw=True)


tab_new, tab_stored = st.tabs(["➕ Add an exam", "🗂️ Stored exams"])

with tab_new:
    with st.form("exam"):
        exam_id = st.text_input("Exam ID", value="cs_midterm", help="A short slug, e.g. cs_midterm")
        sample = st.toggle("Use the sample CS201 midterm pack that ships with the course", value=True)
        c1, c2, c3 = st.columns(3)
        qp = c1.file_uploader("Question paper", type=["pdf"])
        ms = c2.file_uploader("Marking scheme", type=["pdf"])
        me = c3.file_uploader("Methodology", type=["pdf"])
        ma = st.file_uploader("Model answers (optional)", type=["md", "pdf", "txt"])
        if demo():
            st.caption("🧪 Demo mode: a fake 'extractor' always returns the sample midterm. The approval step is real.")
        go = st.form_submit_button("🔎 Extract the exam", type="primary")

    if go:
        files = []
        if not sample:
            for field, upload in (("question_paper", qp), ("marking_scheme", ms), ("methodology", me)):
                if upload:
                    files.append((field, (upload.name, upload.getvalue(), "application/pdf")))
            if ma:
                files.append(("model_answers", (ma.name, ma.getvalue(), ma.type or "text/markdown")))
        with st.spinner("Parsing the PDFs and extracting the exam…"), friendly_errors():
            st.session_state["exam_review"] = client.post(
                "/exams", data={"exam_id": exam_id, "use_sample": sample, "demo": demo()}, files=files
            )

    review = st.session_state.get("exam_review")
    if review:
        summary = review.get("summary") or {}
        status = review["status"]
        colour = {"awaiting_approval": "amber", "stored": "green", "rejected": "red"}[status]
        st.html(
            f"<h4>{escape(summary.get('title', review['exam_id']))} "
            f"{styles.pill(status.replace('_', ' '), colour)}</h4>"
        )
        styles.stat_cards(
            [
                ("Exam ID", review["exam_id"]),
                ("Questions", summary.get("question_count", "–")),
                ("Total marks", f"{summary.get('total_marks', 0):g}"),
                ("Criteria", summary.get("criteria_count", "–")),
            ]
        )
        for problem in review.get("problems") or []:
            st.warning(problem, icon="⚠️")
        if summary.get("questions"):
            show_questions(summary)
        if status == "awaiting_approval":
            st.markdown("#### 🧑‍🏫 Your decision")
            st.caption("Nothing is stored until you approve. 💡 This button resumes a paused LangGraph interrupt.")
            a, r, _ = st.columns([1, 1, 3])
            decision = None
            if a.button("✅ Approve and store", type="primary"):
                decision = True
            if r.button("✋ Reject"):
                decision = False
            if decision is not None:
                with friendly_errors():
                    st.session_state["exam_review"] = client.post(
                        f"/exams/{review['exam_id']}/approve", json_body={"approved": decision, "demo": demo()}
                    )
                st.rerun()
        elif status == "stored":
            st.success("Approved and stored. You can grade against it now.", icon="✅")
            st.page_link("pages/grade.py", label="Grade a batch →", icon="✍️")

with tab_stored:
    with friendly_errors():
        exams = client.get("/exams", demo=demo())
    if exams:
        styles.table(
            ["Exam", "Title", "Questions", "Marks", "Where from"],
            [(e["exam_id"], e["title"], e["question_count"], f"{e['total_marks']:g}", e["source"]) for e in exams],
        )
        chosen = st.selectbox("Look inside", [e["exam_id"] for e in exams])
        with friendly_errors():
            detail = client.get(f"/exams/{chosen}", demo=demo())
        show_questions(detail["summary"])
        with st.expander("The full ExamPaper JSON"):
            st.json(detail["paper"], expanded=False)
