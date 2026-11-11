"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  💬 Week 6 · Day 3: Chat with the results                            │
│                                                                      │
│  What: ask the week 5 results agent anything Asha would ask after a  │
│        marking session. It answers with its four tools: exact stats, │
│        semantic search over graded answers, one student's report,    │
│        and the list of flags.                                        │
│  ⏮ Built on: POST /ask (gradewise/reports/qa_agent.py)              │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import streamlit as st

from gradewise.ui import client, styles
from gradewise.ui.client import ApiError
from gradewise.ui.common import demo, exam_picker

EXAMPLES = [
    "Which question did the class struggle with most?",
    "Why did student_02 lose marks?",
    "Who had trouble with subnetting?",
    "Which answers still need my review?",
]

styles.hero(
    "Chat with results",
    "Numbers come from code, words come from the model. Ask about the class, a question, or one student.",
    kicker="Week 5 · Report Q&A agent",
)

exam_id = exam_picker()
if not exam_id:
    st.stop()

history: list[dict] = st.session_state.setdefault(f"chat-{exam_id}-{demo()}", [])
cols = st.columns(len(EXAMPLES))
pressed = [
    q for col, q in zip(cols, EXAMPLES, strict=True) if col.button(q, width="stretch")
]  # 💡 a list: draw every button
clicked = pressed[0] if pressed else None

for message in history:
    with st.chat_message(message["role"], avatar="🧑‍🏫" if message["role"] == "user" else "🎓"):
        st.markdown(message["content"])

question = st.chat_input("Ask about this exam's results…") or clicked
if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🏫"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🎓"), st.spinner("Thinking (and calling tools)…"):
        try:
            answer = client.post("/ask", json_body={"question": question, "exam_id": exam_id, "demo": demo()})["answer"]
        except ApiError as err:
            answer = f"❌ **{err.title}**: {err.payload.get('what', '')}\n\n💡 {err.payload.get('fix', '')}"
        st.markdown(answer)
    history.append({"role": "assistant", "content": answer})
