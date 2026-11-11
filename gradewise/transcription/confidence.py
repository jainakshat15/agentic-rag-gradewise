"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 4 · Day 4: Confidence, illegible flags, double-pass checks  │
│                                                                      │
│  What: turns three signals into one confidence per answer:           │
│          1. the model's own confidence (cheap, but optimistic)       │
│          2. how much of the text is [illegible]                      │
│          3. whether a SECOND reading agrees with the first (CER)     │
│        and flags anything under the threshold for a human.           │
│  ⏮ Built on: mapper.py, vision.py                                   │
│  ⏭ Used by: graph.py, week5 transcription evaluation (cer / wer)    │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jiwer
from pydantic import BaseModel

from gradewise.schemas.answers import AnswerSheet, TranscribedAnswer

if TYPE_CHECKING:
    from PIL import Image

    from gradewise.schemas.exam import ExamPaper
    from gradewise.transcription.schemas import PageTranscription

ILLEGIBLE = "[illegible]"
REVIEW_THRESHOLD = 0.75  # 💡 below this, a human looks at the answer before it is graded
MAX_PASS_CER = 0.15  # 💡 two readings that differ by more than 15% of characters: someone must check
VERIFY_DPI = 300

_ILLEGIBLE_RE = re.compile(r"\[illegible\]", re.IGNORECASE)


# ── 📏 Error rates (also used by week 5's transcription evaluation) ────────
def _clean(text: str, lowercase: bool) -> str:
    # 💡 Line breaks depend on where the student ran out of paper, not on what they wrote,
    #    so we always collapse whitespace. Case is kept by default: "RAM" vs "ram" is a real difference.
    text = " ".join(text.split())
    return text.lower() if lowercase else text


def cer(reference: str, hypothesis: str, *, lowercase: bool = False) -> float:
    """Character Error Rate: (substitutions + deletions + insertions) / characters in the reference.

    0.0 is perfect. It can go ABOVE 1.0 when the hypothesis is much longer than the reference.
    """
    ref, hyp = _clean(reference, lowercase), _clean(hypothesis, lowercase)
    if not ref:
        # ⚠️ jiwer refuses an empty reference. Blank vs blank is perfect; blank vs anything is 100% wrong.
        return 0.0 if not hyp else 1.0
    return float(jiwer.cer(ref, hyp))


def wer(reference: str, hypothesis: str, *, lowercase: bool = False) -> float:
    """Word Error Rate: the same idea as CER, counted in words."""
    ref, hyp = _clean(reference, lowercase), _clean(hypothesis, lowercase)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.wer(ref, hyp))


# ── 🎯 Scoring ─────────────────────────────────────────────────────────────
def illegible_ratio(text: str) -> float:
    """The share of words that are `[illegible]` markers (0 when the text is empty)."""
    words = _ILLEGIBLE_RE.sub(" ILLEGIBLE ", text).split()
    if not words:
        return 0.0
    return sum(1 for w in words if w == "ILLEGIBLE") / len(words)


def combine_confidence(self_reported: float, illegible: float = 0.0, agreement: float | None = None) -> float:
    """Blend the three signals into one 0-1 score.

    score = self_reported × (1 − min(1, 2 × illegible_ratio)) × agreement

    💡 Why multiply? Each signal can only pull confidence DOWN. A model that says 0.95 but left
       a quarter of the words [illegible] should not get 0.95 (it gets 0.95 × 0.5 = 0.475).
    💡 Why does agreement never push it UP? Two passes of the same model can make the same
       mistake twice ("semafore" read as "semaphore" both times). Agreement is necessary, not
       sufficient, so it is a ceiling, never a bonus.
    """
    score = max(0.0, min(1.0, self_reported))
    score *= 1.0 - min(1.0, 2.0 * max(0.0, illegible))
    if agreement is not None:
        score *= max(0.0, min(1.0, agreement))
    return round(score, 3)


def needs_review(answer: TranscribedAnswer, threshold: float = REVIEW_THRESHOLD) -> bool:
    """The flagging policy: anything below the threshold goes to a human."""
    return answer.confidence < threshold


def flagged_answers(sheet: AnswerSheet, threshold: float = REVIEW_THRESHOLD) -> list[str]:
    """Question IDs a teacher should check before grading."""
    return [a.question_id for a in sheet.answers if needs_review(a, threshold)]


# ── 🔁 Double-pass verification ────────────────────────────────────────────
class VerificationCheck(BaseModel):
    """What happened when we read one low-confidence answer a second time."""

    question_id: str
    pages: list[int]
    first_confidence: float
    second_confidence: float
    cer_between_passes: float
    final_confidence: float
    kept_pass: int  # 1 or 2
    flagged: bool


def double_pass_verify(
    sheet: AnswerSheet,
    pages: Sequence[PageTranscription],
    exam: ExamPaper,
    *,
    llm: Any = None,
    pdf_path: str | Path | None = None,
    images: Sequence[Image.Image] | None = None,
    verify_dpi: int = VERIFY_DPI,
    threshold: float = REVIEW_THRESHOLD,
    max_cer: float = MAX_PASS_CER,
) -> tuple[AnswerSheet, list[VerificationCheck]]:
    """Re-read the pages behind every low-confidence answer, and compare the two readings.

    - With `pdf_path`, the pages are re-rendered at `verify_dpi` (sharper, and pricier);
      otherwise the already-rendered `images` (index = page − 1) are sent again.
    - For each answer: CER between the passes → agreement = 1 − CER. We keep the text of the
      pass that scored higher; confidence = mean of both passes × agreement.
    - If the passes disagree by more than `max_cer`, the answer is flagged for review no matter
      what the model said about itself.

    Returns the updated sheet and one VerificationCheck per re-read answer.
    """
    # 💡 Imported here to avoid a circular import (mapper.py uses this module's scoring).
    from gradewise.transcription.mapper import map_answers
    from gradewise.transcription.pdf_to_images import preprocess, render_page
    from gradewise.transcription.vision import transcribe_page

    low = [a for a in sheet.answers if needs_review(a, threshold) and a.pages]
    if not low:
        return sheet, []
    if pdf_path is None and images is None:
        raise ValueError("double_pass_verify needs either pdf_path or images to re-read pages.")

    question_ids = [q.id for q in exam.questions]
    redo = sorted({p for a in low for p in a.pages})
    by_page = {p.page_number: p for p in pages}
    for page_number in redo:
        if pdf_path is not None:
            # 🚀 PRO: a bigger size cap on purpose. The second pass is where we spend extra pixels.
            image = preprocess(render_page(pdf_path, page_number, dpi=verify_dpi), max_side=2400)
        else:
            image = images[page_number - 1]  # type: ignore[index]
        by_page[page_number] = transcribe_page(image, page_number, question_ids, llm=llm, second_pass=True)

    second = map_answers(sorted(by_page.values(), key=lambda p: p.page_number), exam, sheet.student_id)

    checks: list[VerificationCheck] = []
    updated: dict[str, TranscribedAnswer] = {}
    for first in low:
        other = second.answer_for(first.question_id) or first
        distance = cer(first.text, other.text)
        agreement = 1.0 - min(1.0, distance)
        kept, kept_pass = (other, 2) if other.confidence > first.confidence else (first, 1)
        # 💡 Both readings vote: a confident second pass can't single-handedly "rescue" an answer
        #    the first pass struggled with, and any disagreement pulls the score down.
        final = round((first.confidence + other.confidence) / 2 * agreement, 3)
        flagged = distance > max_cer or final < threshold
        if distance > max_cer:
            # ⚠️ Big disagreement: force the answer under the threshold so needs_review() fires.
            final = min(final, round(threshold - 0.01, 3))
        updated[first.question_id] = kept.model_copy(update={"confidence": final})
        checks.append(
            VerificationCheck(
                question_id=first.question_id,
                pages=first.pages,
                first_confidence=first.confidence,
                second_confidence=other.confidence,
                cer_between_passes=round(distance, 4),
                final_confidence=final,
                kept_pass=kept_pass,
                flagged=flagged,
            )
        )

    answers = [updated.get(a.question_id, a) for a in sheet.answers]
    return sheet.model_copy(update={"answers": answers}), checks
