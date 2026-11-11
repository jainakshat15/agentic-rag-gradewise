"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 4 · Day 4: What a transcribed answer sheet looks like       │
│                                                                      │
│  What: the output of reading a handwritten answer sheet: verbatim    │
│        text per question, where it was found, and how sure we are.   │
│  ⚠️ `text` must be VERBATIM: spelling mistakes preserved. GradeWise  │
│     never "fixes" what the student wrote.                            │
│  ⏭ Used by: gradewise/grading                                       │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscribedAnswer(BaseModel):
    """One student answer, mapped to a question."""

    question_id: str = Field(description='The exam question ID this answer belongs to, e.g. "Q4b"')
    text: str = Field(description="Verbatim transcription, mistakes and all. Empty if not answered.")
    pages: list[int] = Field(default_factory=list, description="1-based page numbers where the answer appears")
    confidence: float = Field(ge=0, le=1, description="0–1: how sure we are the transcription is accurate")
    illegible_spans: list[str] = Field(default_factory=list, description="Words or phrases that could not be read")
    crossed_out: list[str] = Field(default_factory=list, description="Text the student crossed out (not marked)")

    @property
    def is_blank(self) -> bool:
        return not self.text.strip()


class AnswerSheet(BaseModel):
    """All the answers from one student's script."""

    student_id: str
    exam_id: str
    source_file: str
    page_count: int
    answers: list[TranscribedAnswer]
    unmapped_text: list[str] = Field(default_factory=list, description="Text we couldn't link to any question")

    def answer_for(self, question_id: str) -> TranscribedAnswer | None:
        return next((a for a in self.answers if a.question_id == question_id), None)
