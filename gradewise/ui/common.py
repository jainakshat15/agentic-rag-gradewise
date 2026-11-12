"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧩 Week 6 · Day 2–3: Pieces every dashboard page shares             │
│                                                                      │
│  What this file does                                                 │
│      demo()            → is the sidebar's demo toggle on?            │
│      friendly_errors() → shows an ApiError as a what/cause/fix card  │
│      exam_picker()     → choose an exam (remembered across pages)    │
│      live_progress()   → per-student progress bars + an event log,   │
│                          fed by the batch's SSE stream (week 6 day 3)│
│                                                                      │
│  ⏭ Used by: gradewise/ui/pages/*.py                                 │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from html import escape
from typing import Any

import streamlit as st

from gradewise.ui import client, styles
from gradewise.ui.client import ApiError

ICONS = {
    "queued": "🗂️",
    "load_exam": "📝",
    "transcribe": "✍️",
    "review_transcription": "🔎",
    "grade": "🧮",
    "grade_question": "·",
    "aggregate": "📊",
    "human_review": "🧑‍🏫",
    "write_report": "📄",
    "store": "💾",
    "paused": "⏸️",
    "resumed": "▶️",
    "done": "✅",
    "failed": "❌",
}


def demo() -> bool:
    """The sidebar toggle (app.py). Every API call passes it, so one switch flips the whole dashboard."""
    return bool(st.session_state.get("demo_mode", True))


def show_error(err: ApiError) -> None:
    p = err.payload
    lines = [f"**{p.get('error', 'Error')}**: {p.get('what', '')}"]
    if p.get("cause"):
        lines.append(f"- **Likely cause:** {p['cause']}")
    if p.get("fix"):
        lines.append(f"- **Fix:** {p['fix']}")
    if p.get("help"):
        lines.append(f"- **Help:** `{p['help']}`")
    st.error("\n".join(lines), icon="❌")


@contextmanager
def friendly_errors(stop: bool = True) -> Iterator[None]:
    """`with friendly_errors(): ...`: an API error becomes a red card instead of a stack trace."""
    try:
        yield
    except ApiError as err:
        show_error(err)
        if stop:
            st.stop()


def exam_picker(label: str = "Exam", key: str = "exam_id") -> str | None:
    """A selectbox of every exam the API knows (approved ones, plus the bundled sample)."""
    with friendly_errors():
        exams = client.get("/exams", demo=demo())
    if not exams:
        st.info("No exams yet. Add one on the 📝 Exams page.", icon="📝")
        return None
    ids = [e["exam_id"] for e in exams]
    names = {
        e["exam_id"]: f"{e['exam_id']} · {e['title']}" + (" (sample)" if e["source"] == "sample" else "") for e in exams
    }
    current = st.session_state.get(key)
    index = ids.index(current) if current in ids else 0
    choice = st.selectbox(label, ids, index=index, format_func=names.get)
    st.session_state[key] = choice
    return choice


def _bar_text(p: dict[str, Any]) -> str:
    stage = p.get("stage", "queued").replace("_", " ")
    return f"**{p['student_id']}** · {stage} · {p.get('questions_done', 0)}/{p.get('questions_total', 0)} questions"


def live_progress(batch_id: str) -> str:
    """Follow a batch live until it pauses, finishes or fails. Returns that final status.

    💡 This is the whole "streaming from LangGraph to the UI" story: grade_batch(on_event=...) → the job's
       event list → the SSE endpoint → httpx.stream here → placeholders that we overwrite in place.
    """
    status = client.get(f"/batches/{batch_id}")
    progress = {p["student_id"]: p for p in status["progress"]}
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown("##### 👩‍🎓 Students")
        bars = {sid: st.empty() for sid in progress}
    with right:
        st.markdown("##### 📡 Event log")
        log_box = st.empty()
    # 💡 Remember what's already shown, so following the batch again (after a review) only adds new lines.
    lines: list[str] = st.session_state.setdefault(f"log-{batch_id}", [])
    after = st.session_state.get(f"last-{batch_id}", 0)

    def draw() -> None:
        for sid, p in progress.items():
            bars[sid].progress(min(100, int(p.get("percent", 0))) / 100, text=_bar_text(p))
        log_box.html(f'<div class="gw-log">{"<br>".join(lines[-40:]) or "waiting for events…"}</div>')

    draw()
    final = status["status"]
    for name, event in client.stream_events(batch_id, after=after):
        if name == "end":
            final = event["status"]
            break
        st.session_state[f"last-{batch_id}"] = event["id"]
        if event.get("progress"):
            progress[event["student_id"]] = event["progress"]
        stage = event["stage"]
        stamp = datetime.fromisoformat(event["time"]).astimezone().strftime("%H:%M:%S")
        who = f' <span class="who">{escape(event["student_id"])}</span>' if event.get("student_id") else ""
        lines.append(
            f'<span class="t">{stamp}</span> {ICONS.get(stage, "▸")} <span class="stage">{escape(stage)}</span>'
            f"{who} {escape(event.get('message', ''))}"
        )
        if stage == "done":
            for p in progress.values():
                p.update(percent=100, stage="done")
        draw()
    draw()
    return final


def status_pill(status: str) -> str:
    colour = {"done": "green", "paused": "amber", "running": "cyan", "queued": "muted", "failed": "red"}
    return styles.pill(status, colour.get(status, "muted"))
