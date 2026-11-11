"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 4 · Day 4: Mapping answers to questions                     │
│                                                                      │
│  What: pure Python (no LLM!) that turns per-page segments into ONE   │
│        TranscribedAnswer per exam question. Handles "contd." across  │
│        pages, answers out of order, missing answers, and labels      │
│        like "4(b)", "Ans 4 b", "Q.4b" → "Q4b".                       │
│  💡 Let the model READ; let plain code DECIDE. Deterministic logic   │
│     is testable, free, and never invents an answer.                  │
│  ⏭ Used by: graph.py, confidence.py                                 │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from gradewise.schemas.answers import AnswerSheet, TranscribedAnswer
from gradewise.schemas.exam import ExamPaper
from gradewise.transcription.confidence import combine_confidence, illegible_ratio
from gradewise.transcription.schemas import PageTranscription

# 💡 Words students use to say "this carries on from before".
_CONTD_RE = re.compile(r"\b(?:contd|cont'd|cont|ctd|continued|continuation|continues)\b\.?|→|->|\.\.\.", re.I)
_WORD_PREFIX_RE = re.compile(
    r"\b(?:answers?|ans)\b\.?\s+(?:to|for|of)\b|\b(?:question|ques|qn|qno|answer|ans|no)\b\.?", re.I
)  # 💡 "Answer to 4b", "Ans. 4(ii)", "Question 10:" all lose their words before we read the number
_LABEL_RE = re.compile(
    r"""^\s*
        q?\s*[.:#\-]?\s*            # optional leading Q (after the words above are removed)
        (?P<num>\d{1,2})            # question number
        \s*[.:\-]?\s*
        (?P<parts>(?:\(?\s*[a-z]{1,4}\s*\)?\s*[.:\-]?\s*){0,3})   # sub-parts: b, (b), (ii), (b)(i)
        \s*[.:)\-]*\s*$""",
    re.I | re.X,
)
_PART_ONLY_RE = re.compile(r"^\s*\(?\s*(?P<part>[a-z]|i{1,3}|iv|v|vi{0,3})\s*\)\s*[.:]?\s*$", re.I)
_PART_RE = re.compile(r"[a-z]{1,4}", re.I)
_ROMAN = {"i": "a", "ii": "b", "iii": "c", "iv": "d", "v": "e", "vi": "f", "vii": "g", "viii": "h"}
_STRUCK_RE = re.compile(r"~~(.+?)~~", re.S)  # ⚠️ belt and braces: some models mark strike-through this way


def _canonical(num: str, part: str | None) -> str:
    return f"Q{int(num)}{(part or '').lower()}"


def _resolve(num: str, part: str | None, valid: set[str] | None, *, allow_parent: bool = True) -> str | None:
    """Pick the exam ID for a number + sub-part, trying roman numerals (ii → b) if needed.

    `allow_parent=False` refuses the "4(a) → Q4" fallback (used for a bare "(a)" label).
    """
    candidate = _canonical(num, part)
    if valid is None or candidate in valid:
        return candidate
    if part and part.lower() in _ROMAN:
        roman = _canonical(num, _ROMAN[part.lower()])
        if roman in valid:
            return roman
    if part and allow_parent and _canonical(num, None) in valid:
        # 💡 The exam has no sub-parts for this question, but the student split their answer: "4(a)" → Q4.
        return _canonical(num, None)
    return None


def _resolve_parts(num: str, parts: list[str], valid: set[str] | None) -> str | None:
    """Several sub-parts ("4(b)(i)"): take the DEEPEST one the exam knows, Q4bi → Q4b → Q4."""
    if not parts:
        return _resolve(num, None, valid)
    if valid is not None:
        for depth in range(len(parts), 1, -1):
            candidate = _canonical(num, "".join(parts[:depth]))
            if candidate in valid:
                return candidate
    # 💡 our exam IDs carry at most one sub-part letter, so without an exam "4(b)(i)" → Q4b
    return _resolve(num, parts[0], valid)


def is_continuation_label(label: str | None) -> bool:
    """True for no label at all, or a label that only says "contd." (no question number)."""
    if label is None or not label.strip():
        return True
    if re.search(r"\d", label):
        return False
    return bool(_CONTD_RE.search(label)) or not _WORD_PREFIX_RE.sub("", label).strip(" .:-()")


def normalise_label(
    label: str | None, valid_ids: Iterable[str] | None = None, *, current: str | None = None
) -> str | None:
    """Turn a label as written into an exam question ID, or None if it can't be matched.

    >>> normalise_label("4(b)")
    'Q4b'
    >>> normalise_label("Answer to 4b")
    'Q4b'
    >>> normalise_label("4(b)(i)")   # nested sub-parts: the deepest ID the exam knows
    'Q4b'
    >>> normalise_label("(c)", current="Q7a")   # a bare sub-part continues the current question number
    'Q7c'

    With `valid_ids`, the result must be one of them (so "Q4b" on an exam that has no 4(b) → None).
    ⚠️ A bare sub-part like "(a)" is only matched when the current question HAS sub-parts. If the
       exam's Q1 has none, "(a)" after "Q1 contd." is something else (rough work?), so → None.
    """
    if label is None:
        return None
    valid = set(valid_ids) if valid_ids is not None else None
    text = _CONTD_RE.sub(" ", label)
    text = _WORD_PREFIX_RE.sub(" ", text).strip()

    match = _LABEL_RE.match(text)
    if match:
        return _resolve_parts(match["num"], _PART_RE.findall(match["parts"] or ""), valid)

    part_only = _PART_ONLY_RE.match(text)
    if part_only and current:
        number = re.match(r"Q(\d+)", current)
        if number:
            return _resolve(number[1], part_only["part"], valid, allow_parent=False)
    return None


@dataclass
class _Bucket:
    """Everything collected for one question while we walk the pages."""

    texts: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    weighted_conf: float = 0.0
    weight: float = 0.0
    crossed_out: list[str] = field(default_factory=list)
    illegible: list[str] = field(default_factory=list)

    def add(self, text: str, page: int, confidence: float, crossed: list[str], illegible: list[str]) -> None:
        if text.strip():
            self.texts.append(text.strip())
        if page not in self.pages:
            self.pages.append(page)
        # 💡 Weight by length: a confident 3-word "contd." shouldn't outvote a shaky 200-word answer.
        w = max(1, len(text))
        self.weighted_conf += confidence * w
        self.weight += w
        self.crossed_out.extend(crossed)
        self.illegible.extend(illegible)

    def to_answer(self, question_id: str) -> TranscribedAnswer:
        text = "\n".join(self.texts)
        self_reported = self.weighted_conf / self.weight if self.weight else 0.0
        return TranscribedAnswer(
            question_id=question_id,
            text=text,
            pages=sorted(self.pages),
            confidence=combine_confidence(self_reported, illegible_ratio(text)),
            illegible_spans=self.illegible,
            crossed_out=self.crossed_out,
        )


def _split_struck(text: str) -> tuple[str, list[str]]:
    struck = [s.strip() for s in _STRUCK_RE.findall(text)]
    return _STRUCK_RE.sub("", text), struck


def map_answers(
    pages: Sequence[PageTranscription],
    exam: ExamPaper,
    student_id: str,
    *,
    source_file: str = "",
    page_count: int | None = None,
    blank_confidence: float = 1.0,
) -> AnswerSheet:
    """Merge every page's segments into an AnswerSheet with exactly one answer per exam question.

    Rules, in order:
      1. Pages are processed in page order; segments in the order the model listed them.
      2. A segment with a recognisable label starts (or re-opens) that question.
      3. An unlabelled or "contd." segment is appended to the question we were last in,
         even across a page break. With no question open yet, it goes to `unmapped_text`.
      4. A label we can't match to the exam ("Q11", "rough work") → `unmapped_text`, and
         following unlabelled text goes there too (we don't guess).
      5. Crossed-out text is kept in `crossed_out`, never in `text` (it is not marked).
      6. Questions nobody answered get an EMPTY answer: reported, never invented.
         If any answer-like segment ended up unmapped, blanks get a lower confidence so a human
         checks whether the "missing" answer is actually hiding in the unmapped text.
      7. `other_text` (names, headers, margin notes) goes to `unmapped_text` too, verbatim.
    """
    valid = [q.id for q in exam.questions]
    valid_set = set(valid)
    buckets: dict[str, _Bucket] = {}
    unmapped: list[str] = []
    unmapped_answers = 0  # answer-like segments we couldn't place (headers / names don't count)
    current: str | None = None  # the question we are "inside" right now
    context: str | None = None  # the last question NUMBER we saw, e.g. "Q7", so "(b)" can become "Q7b"
    lost = False  # True after an unknown label: following continuations are unmapped too

    for page in sorted(pages, key=lambda p: p.page_number):
        for seg in page.segments:
            text, struck = _split_struck(seg.text)
            crossed = [*seg.crossed_out, *struck]
            target: str | None
            number = re.search(r"\d{1,2}", seg.label or "")
            if number:
                context = f"Q{int(number[0])}"
            if seg.label and not is_continuation_label(seg.label):
                target = normalise_label(seg.label, valid_set, current=context)
                lost = target is None
            elif seg.label and normalise_label(seg.label, valid_set, current=context):
                # e.g. "(c) contd.": no number, but a sub-part of the question we are in
                target = normalise_label(seg.label, valid_set, current=context)
                lost = False
            else:
                target = None if lost else current

            if target is None:
                if text.strip():
                    prefix = f"[page {page.page_number}" + (f", label {seg.label!r}]" if seg.label else "]")
                    unmapped.append(f"{prefix} {text.strip()}")
                    unmapped_answers += 1
                continue
            current = target
            buckets.setdefault(target, _Bucket()).add(
                text, page.page_number, seg.confidence, crossed, list(seg.illegible_spans)
            )
        for other in page.other_text:
            if other.strip():
                unmapped.append(f"[page {page.page_number}, not an answer] {other.strip()}")

    # ⚠️ A "missing" answer may really be unmapped text. Don't be 100% sure it's blank.
    blank_conf = blank_confidence if not unmapped_answers else min(blank_confidence, 0.5)
    answers = [
        buckets[qid].to_answer(qid)
        if qid in buckets
        else TranscribedAnswer(question_id=qid, text="", pages=[], confidence=blank_conf)
        for qid in valid
    ]
    return AnswerSheet(
        student_id=student_id,
        exam_id=exam.exam_id,
        source_file=source_file,
        page_count=page_count if page_count is not None else len(pages),
        answers=answers,
        unmapped_text=unmapped,
    )
