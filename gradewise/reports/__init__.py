"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧱 Week 5 · Day 3: gradewise.reports, report cards                  │
│                                                                      │
│  What: graded answers → report cards people read, store and query.   │
│      writer.py   → feedback + HTML + PDF report cards                │
│      store.py    → SQLite rows + a Chroma `reports_<exam>` index     │
│      qa_agent.py → the "chat with results" agent                     │
│  ⏮ Built on: gradewise/grading   ⏭ Used by: orchestrator.py, wk 6   │
╰──────────────────────────────────────────────────────────────────────╯
"""

from gradewise.reports.store import (
    get_report,
    index_report,
    list_reports,
    report_documents,
    reports_collection,
    save_report,
    search_reports,
    store_report,
)
from gradewise.reports.writer import (
    ReportFeedback,
    render_html,
    render_pdf,
    rule_based_feedback,
    save_report_files,
    write_feedback,
)

__all__ = [
    "ReportFeedback",
    "ask",
    "build_report_agent",
    "get_report",
    "index_report",
    "list_reports",
    "render_html",
    "render_pdf",
    "report_documents",
    "reports_collection",
    "rule_based_feedback",
    "save_report",
    "save_report_files",
    "search_reports",
    "store_report",
    "write_feedback",
]


def __getattr__(name: str):
    # 💡 Lazy: the Q&A agent pulls in pandas and the agent stack, which writing a report doesn't need.
    if name in ("ask", "build_report_agent"):
        from gradewise.reports import qa_agent

        return getattr(qa_agent, name)
    raise AttributeError(f"module 'gradewise.reports' has no attribute {name!r}")
