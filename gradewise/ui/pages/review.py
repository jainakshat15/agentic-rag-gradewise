"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🔍 Week 6 · Day 2: The review queue, where Emma has the last word   │
│                                                                      │
│  What: every batch that paused for a human. For each flagged answer  │
│        you see the question, what the student actually wrote, why    │
│        it was flagged, and the grader's criterion-by-criterion       │
│        justification with its evidence. Override any mark, approve,  │
│        and the batch resumes (LangGraph `Command(resume=...)`).      │
│  ⏮ Built on: GET /batches?status=paused, POST /batches/{id}/review  │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from html import escape

import streamlit as st

from gradewise.ui import client, styles
from gradewise.ui.common import demo, friendly_errors, live_progress

styles.hero(
    "Review queue",
    "Principle 7: an uncertain mark is a question for a teacher, not a decision for a model.",
    kicker="Human in the loop",
)

with friendly_errors():
    queue = [b for b in client.get("/batches", status="paused") if b["demo"] == demo()]

resumed = st.session_state.pop("resumed", None)
if resumed:
    st.markdown(f"#### ▶️ Resuming `{resumed}`…")
    with friendly_errors():
        final = live_progress(resumed)
    if final == "done":
        st.success("✅ Batch complete: report cards written and stored.", icon="🎉")
        st.page_link("pages/reports.py", label="See the reports →", icon="📊")
    elif final == "paused":
        st.info("The next student is waiting below.", icon="⏸️")
    with friendly_errors():
        queue = [b for b in client.get("/batches", status="paused") if b["demo"] == demo()]

if not queue:
    st.success("Nothing is waiting for you. 🎉 Every graded answer has been approved.", icon="✅")
    st.stop()

labels = {
    b["batch_id"]: f"{b['batch_id']} · {b['pending']['student_id']} · {b['pending']['type'].replace('_', ' ')}"
    for b in queue
}
batch_id = st.selectbox("Paused batch", list(labels), format_func=labels.get)
batch = next(b for b in queue if b["batch_id"] == batch_id)
pending = batch["pending"]
is_marks = pending["type"] == "human_review"

styles.stat_cards(
    [
        ("Student", pending["student_id"]),
        ("Waiting for", "mark review" if is_marks else "transcription check"),
        ("Flagged items", len(pending["items"])),
        ("Current total", f"{pending['total']:g}/{pending['max_total']:g}" if is_marks else "not graded yet"),
    ]
)

with st.form(f"review-{batch_id}-{pending['student_id']}"):
    overrides: dict[str, float] = {}
    corrections: dict[str, str] = {}
    for item in pending["items"]:
        qid = item["question_id"]
        with st.container(border=True):
            reason = item.get("reason") or "low transcription confidence"
            head = f"<b>{escape(qid)}</b> "
            if is_marks:
                head += styles.pill(f"{item['awarded']:g}/{item['max_marks']:g} marks", "violet")
            else:
                head += styles.pill(f"confidence {item.get('confidence') or 0:.2f}", "amber")
            head += styles.pill(reason[:90], "amber")
            st.html(
                f"<div>{head}</div><div class='gw-muted' style='margin-top:4px'>{escape(item['question_text'])}</div>"
            )

            left, right = st.columns(2, gap="large")
            with left:
                st.markdown("**✍️ What the student wrote**")
                st.html(f'<div class="gw-answer">{escape(item["answer_text"]) or "<i>(blank)</i>"}</div>')
                if item.get("illegible_spans"):
                    st.caption("Couldn't read: " + ", ".join(item["illegible_spans"]))
            with right:
                if is_marks:
                    st.markdown("**🧮 The grader's reasoning**")
                    cards = []
                    for c in item["criteria"]:
                        marks = f"{c['awarded']:g}/{c['max_marks']:g}" if c.get("max_marks") else f"{c['awarded']:g}"
                        evidence = "".join(styles.pill(e[:70], "cyan") for e in c.get("evidence", []))
                        cards.append(
                            f'<div class="gw-crit"><div class="top"><span>{escape(c["criterion_id"])}</span>'
                            f"<span>{marks}</span></div><div class='gw-muted'>{escape(c['description'])}</div>"
                            f'<div class="why">{escape(c["justification"])}</div><div>{evidence}</div></div>'
                        )
                    st.html("".join(cards) or "<div class='gw-muted'>No criterion breakdown.</div>")
                    if item.get("feedback"):
                        st.caption(f"Feedback to the student: {item['feedback']}")
                else:
                    st.markdown("**🔎 Correct the transcription** (type exactly what the student wrote)")

            if is_marks:
                value = st.number_input(
                    f"Marks for {qid}",
                    min_value=0.0,
                    max_value=float(item["max_marks"]),
                    value=float(item["awarded"]),
                    step=0.5,
                    key=f"mark-{batch_id}-{pending['student_id']}-{qid}",
                )
                if value != item["awarded"]:
                    overrides[qid] = value
            else:
                text = st.text_area(
                    f"Transcription of {qid}",
                    value=item["answer_text"],
                    key=f"text-{batch_id}-{pending['student_id']}-{qid}",
                )
                if text != item["answer_text"]:
                    corrections[qid] = text

    approve = st.form_submit_button("✅ Approve and continue", type="primary")

if approve:
    body = {"student_id": pending["student_id"], "overrides": overrides, "corrections": corrections, "approved": True}
    with friendly_errors():
        client.post(f"/batches/{batch_id}/review", json_body=body)
    changed = len(overrides) + len(corrections)
    st.toast(f"Approved {pending['student_id']} with {changed} change(s).", icon="✅")
    st.session_state["resumed"] = batch_id
    st.rerun()
