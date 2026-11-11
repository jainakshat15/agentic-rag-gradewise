"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📊 Week 6 · Day 3: Reports, the class and each student              │
│                                                                      │
│  What: the class overview (KPIs, mark distribution, hardest          │
│        questions), then any student's report card exactly as the     │
│        student would see it, with a PDF download.                    │
│  ⏮ Built on: GET /reports/{exam}, /reports/{exam}/{sid}?format=...  │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import base64

import streamlit as st

from gradewise.ui import client, styles
from gradewise.ui.common import demo, exam_picker, friendly_errors

styles.hero(
    "Reports",
    "How the class did, which questions hurt, and every student's report card with reasons and evidence.",
    kicker="Week 5 · Report cards",
)

exam_id = exam_picker()
if not exam_id:
    st.stop()
with friendly_errors():
    overview = client.get(f"/reports/{exam_id}", demo=demo())

if not overview["count"]:
    st.info("No report cards for this exam yet. Grade a batch first.", icon="✍️")
    st.page_link("pages/grade.py", label="Grade a batch →", icon="✍️")
    st.stop()

styles.stat_cards(
    [
        ("Students", overview["count"]),
        ("Class mean", f"{overview['mean_percent']:g}%"),
        ("Median", f"{overview['median_percent']:g}%"),
        ("Still flagged", overview["flagged"], "answers awaiting review"),
    ]
)

left, right = st.columns(2, gap="large")
with left:
    st.markdown("##### 📈 Mark distribution")
    buckets = ["0–39", "40–54", "55–69", "70–84", "85–100"]
    edges = [40, 55, 70, 85, 101]
    counts = [0] * len(buckets)
    for s in overview["students"]:
        counts[next(i for i, edge in enumerate(edges) if s["percentage"] < edge)] += 1
    styles.bar_chart(buckets, counts, colours=[styles.RED, styles.AMBER, styles.VIOLET, styles.CYAN, styles.GREEN])
    st.caption("Grade bands: F < 40 ≤ D < 55 ≤ C < 70 ≤ B < 85 ≤ A")
with right:
    st.markdown("##### 🧗 Hardest questions (mean %)")
    hardest = overview["questions"][:6]
    styles.bar_chart(
        [q["question_id"] for q in hardest], [q["mean_percent"] for q in hardest], max_value=100, suffix="%"
    )
    st.caption("💡 These numbers come from code (exact), never from a model's arithmetic.")

st.markdown("##### 👩‍🎓 Students")
styles.table(
    ["Student", "Total", "%", "Band", "Review"],
    [
        (
            f"<b>{s['student_id']}</b>",
            f"{s['total']:g}/{s['max_total']:g}",
            f"{s['percentage']:g}",
            styles.pill(s["grade_band"], styles.band_colour(s["grade_band"])),
            styles.pill("flagged", "amber") if s["needs_review"] else styles.pill("approved", "green"),
        )
        for s in overview["students"]
    ],
    raw=True,
)

st.divider()
st.markdown("#### 📄 Report card")
ids = [s["student_id"] for s in overview["students"]]
student = st.selectbox("Student", ids)


@st.cache_data(ttl=30, show_spinner=False)
def fetch(exam: str, sid: str, fmt: str, demo_mode: bool) -> bytes:
    return client.get_bytes(f"/reports/{exam}/{sid}", format=fmt, demo=demo_mode)


with friendly_errors():
    html = fetch(exam_id, student, "html", demo()).decode("utf-8")
    pdf = fetch(exam_id, student, "pdf", demo())
st.download_button(
    "⬇️ Download PDF", data=pdf, file_name=f"{exam_id}_{student}.pdf", mime="application/pdf", type="primary"
)
# 💡 An iframe keeps the card's own CSS from leaking into the dashboard. A data: URL gives it an opaque origin,
#    so nothing inside (the card holds model-written feedback) can ever script the dashboard around it.
st.iframe("data:text/html;base64," + base64.b64encode(html.encode("utf-8")).decode("ascii"), height=1100)
