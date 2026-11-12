"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧱 Week 4: gradewise.transcription, reading handwriting             │
│                                                                      │
│  What: handwritten answer-sheet PDF → AnswerSheet                    │
│      pdf_to_images.py → render, clean up, encode page images  (d3)   │
│      vision.py        → verbatim transcription of one page    (d4)   │
│      mapper.py        → segments → one answer per question    (d4)   │
│      confidence.py    → scores, CER/WER, double-pass checks   (d4)   │
│      graph.py         → the LangGraph subgraph                (d5)   │
│  ⏭ Used by: week4/project/transcription_graph.py, week5 grading     │
╰──────────────────────────────────────────────────────────────────────╯
"""

from gradewise.transcription.confidence import (
    VerificationCheck,
    cer,
    combine_confidence,
    double_pass_verify,
    flagged_answers,
    illegible_ratio,
    needs_review,
    wer,
)
from gradewise.transcription.graph import build_transcription_graph, transcribe_sheet
from gradewise.transcription.mapper import map_answers, normalise_label
from gradewise.transcription.pdf_to_images import image_block, preprocess, render_page, render_pages, to_data_url
from gradewise.transcription.schemas import AnswerSegment, PageTranscription
from gradewise.transcription.vision import transcribe_page

__all__ = [
    "AnswerSegment",
    "PageTranscription",
    "VerificationCheck",
    "build_transcription_graph",
    "cer",
    "combine_confidence",
    "double_pass_verify",
    "flagged_answers",
    "illegible_ratio",
    "image_block",
    "map_answers",
    "needs_review",
    "normalise_label",
    "preprocess",
    "render_page",
    "render_pages",
    "to_data_url",
    "transcribe_page",
    "transcribe_sheet",
    "wer",
]
