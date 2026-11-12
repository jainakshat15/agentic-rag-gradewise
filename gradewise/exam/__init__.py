"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Weeks 1 & 3: Exam understanding                                  │
│                                                                      │
│  What this package does                                              │
│    Emma's exam pack (question paper + marking scheme + methodology)  │
│    becomes ONE validated ExamPaper that every later stage grades     │
│    against.                                                          │
│                                                                      │
│      parse.py    → PDF → clean Markdown, headers/footers stripped    │
│      extract.py  → one LLM call (wk 1) + self-correcting retry (wk 3)│
│      graph.py    → the LangGraph workflow with admin approval (wk 3) │
│      summary.py  → a pretty table of what was extracted              │
│                                                                      │
│  ⏭ Used by: gradewise/grading (week 5), the API and dashboard (wk 6) │
╰──────────────────────────────────────────────────────────────────────╯
"""

from gradewise.exam.extract import ExamExtractionError, exam_pack, extract_exam, extract_once, extract_with_retry
from gradewise.exam.parse import ParsedDocument, ParsedPage, parse_pdf, read_document
from gradewise.exam.summary import exam_summary, print_exam_summary

# 💡 graph.py is imported on demand (`from gradewise.exam.graph import build_exam_graph`)
#    so that week 1 code doesn't need to load LangGraph.

__all__ = [
    "ExamExtractionError",
    "ParsedDocument",
    "ParsedPage",
    "exam_pack",
    "exam_summary",
    "extract_exam",
    "extract_once",
    "extract_with_retry",
    "parse_pdf",
    "print_exam_summary",
    "read_document",
]
