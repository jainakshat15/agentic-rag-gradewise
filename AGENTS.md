# AGENTS.md: guidance for AI coding assistants

> 👋 **Learner:** this file is for the AI assistant in your editor (Cursor, Claude Code, Copilot…). It helps the assistant understand this repo so it gives you answers that fit the course. You don't need to read it, but you're welcome to!

## What this repo is
The companion repo for the Udemy course **"Agentic RAG with LangChain & LangGraph"**. Over 6 weeks, learners build **GradeWise**: an agentic RAG system that ingests course PDFs, converts an exam (question paper + marking scheme + methodology) into JSON, transcribes handwritten answer sheets with a vision LLM, grades each answer with rubric + retrieved evidence, and writes report cards.

## Layout
- `weekN/dayN.ipynb`: the lesson notebooks, in teaching order. `weekN/solutions/` holds the reference solutions; `weekN/project/` (and `week2/implementation`, `pro_implementation`, `evaluation`) holds that week's standalone milestone code.
- `gradewise/`: the finished package. `config.py` (settings from `.env`), `console.py` (rich output), `errors.py` (friendly errors), `models.py` (model factory).
- `setup/`: per-OS setup guides, `diagnostics.py`, `troubleshooting.ipynb`.
- `data/`: the course knowledge base, exam pack, handwritten answer sheets, golden marks.

## Conventions you should follow when helping
1. **Python 3.12 + uv.** Run code with `uv run …`; add packages with `uv add …`. Never suggest `pip install` or editing `uv.lock`.
2. **LangChain 1.x APIs only.** Use `langchain.agents.create_agent` (not the deprecated `langgraph.prebuilt.create_react_agent`), `init_chat_model("provider:model")`, `response_format=` for structured output, and middleware from `langchain.agents.middleware`. Use LangGraph's `StateGraph`, `Send`, `interrupt`, `Command` and checkpointers.
3. **Never hard-code model names or keys.** Get models from `gradewise.models` (`chat_model()`, `grading_model()`, `vision_model()`, `embeddings()`); settings come from `gradewise.config.settings`.
4. **Keep the course voice.** New `.py` files start with the boxed banner (week · day · what · run · built on / used by). Use `gradewise.console` (`success`, `step`, `tip`, `warn`) for output and subclasses of `gradewise.errors.GradeWiseError` for errors.
5. **Comment legend:** `# 💡` concept · `# ⚠️` pitfall · `# 🔁 RECAP` link back · `# 🧪 TRY` experiment · `# 🚀 PRO` production tip · `# ✋ TODO` learner exercise.
6. **Cross-platform.** Use `pathlib`, UTF-8 file I/O, and no shell-specific commands in Python. Learners are on Windows, Mac and Linux.
7. **Faithfulness matters.** Handwriting transcription must stay verbatim (never correct the student). Grades must never exceed a question's max marks, and uncertain cases are flagged for human review.
8. **Don't spoil exercises.** If the learner is working in a `✋ TODO` cell, give hints before full solutions unless asked.

## Useful commands
```bash
uv sync                          # install / update the environment
uv run setup/diagnostics.py      # check the machine is ready
uv run python -m gradewise info  # show current settings
uv run pytest                    # run the tests
uv run jupyter lab               # notebooks in the browser
```
