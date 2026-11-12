"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧱 Week 5 · Day 3: Chat with the results                            │
│                                                                      │
│  What: an interactive Q&A loop over the stored report cards.         │
│        Ask anything Emma would ask after a marking session.          │
│  Run:  uv run week5/project/report_qa.py                             │
│        uv run week5/project/report_qa.py -q "Which question was the  │
│               hardest?"                                              │
│  ⏮ Built on: gradewise/reports/qa_agent.py (4 tools, create_agent)  │
│  ⏮ Needs:    graded reports: uv run week5/project/orchestrator.py   │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.markdown import Markdown

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gradewise.console import banner, console, tip  # noqa: E402
from gradewise.db import list_reports  # noqa: E402
from gradewise.errors import GradeWiseError  # noqa: E402
from gradewise.reports.qa_agent import NoReportsError, ask, build_report_agent  # noqa: E402

EXAMPLES = [
    "Which question did the class struggle with most?",
    "Why did student_04 lose marks on Q7a?",
    "Who had trouble with subnetting?",
    "Which answers still need my review?",
]


def main(
    question: Annotated[str | None, typer.Option("--question", "-q", help="Ask one question and exit")] = None,
    exam_id: Annotated[str, typer.Option("--exam", help="Exam ID")] = "cs_midterm",
) -> None:
    banner("Week 5 · Chat with results", f"ask GradeWise about exam {exam_id}")
    try:
        # 💡 Build the agent once and reuse it for every question in the loop (a missing key fails here).
        agent = build_report_agent(exam_id)
        reports = list_reports(exam_id)
        if not reports:
            raise NoReportsError(f'No report cards are stored for exam "{exam_id}".')
        console.print(f"[muted]{len(reports)} report card(s) loaded.[/]")
        if question:
            console.print(Markdown(ask(question, exam_id, agent=agent)))
            return
        tip("Try: " + "  ·  ".join(EXAMPLES))
        while True:
            asked = typer.prompt("\n🧑‍🏫 You", default="", show_default=False).strip()
            if asked.lower() in ("", "exit", "quit", "q"):
                break
            with console.status("[muted]thinking…[/]"):
                answer = ask(asked, exam_id, agent=agent)
            console.print("[primary]🎓 GradeWise[/]")
            console.print(Markdown(answer))
    except GradeWiseError as err:
        err.show()
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    typer.run(main)
