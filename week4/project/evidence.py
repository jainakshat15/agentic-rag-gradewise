"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 4 · Day 2: The evidence retriever, the demo                 │
│                                                                      │
│  What this file does                                                 │
│    Takes one exam question (with its marking criteria) and Sam's     │
│    answer, and shows the course-note passages the week 5 grader      │
│    will mark against, with a citation for each.                      │
│    No LLM is called: only the embedding model, for the search.       │
│                                                                      │
│  Run : uv run week4/project/evidence.py                              │
│  ⏮ Built on: gradewise/rag/evidence.py                               │
│  ⏭ Used by: week5 grading agent (`retrieve_reference` tool)          │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gradewise.console import banner, console, success, tip  # noqa: E402
from gradewise.errors import GradeWiseError  # noqa: E402
from gradewise.rag import find_evidence  # noqa: E402
from gradewise.schemas import MarkingCriterion, Question  # noqa: E402

# 💡 A question exactly as week 3's exam extractor produces it.
QUESTION = Question(
    id="Q4",
    number="4",
    text="Define deadlock and state the four necessary conditions for it to occur.",
    max_marks=5,
    question_type="definition",
    criteria=[
        MarkingCriterion(
            id="Q4.c1",
            description="Defines deadlock: a set of processes each waiting for a resource held by another",
            marks=1,
            keywords=["blocked", "waiting", "held by another"],
        ),
        MarkingCriterion(
            id="Q4.c2",
            description="Names the four Coffman conditions",
            marks=4,
            keywords=["mutual exclusion", "hold and wait", "no pre-emption", "circular wait"],
            partial_credit_rule="1 mark per correct condition",
        ),
    ],
    methodology="Award marks positively. Spelling mistakes are fine if the term is clear.",
    reference_topics=["deadlock", "Coffman conditions"],
)

# ⚠️ Sam's answer is partly WRONG (starvation is not a deadlock condition). Watch that the
#    evidence still comes from the deadlock notes, not from whatever Sam happened to write.
SAM_ANSWER = "Deadlock is when a process waits forever. Conditions: mutual exclusion, starvation, hold and wait."


def main() -> None:
    banner("Week 4 · Evidence retriever", f"{QUESTION.id}: {QUESTION.text}")
    evidence = find_evidence(QUESTION, SAM_ANSWER, k=4)
    for rank, doc in enumerate(evidence, start=1):
        console.print(f"[primary]{rank}. {doc.metadata['citation']}[/]")
        console.print(f"   [muted]found by: {'; '.join(doc.metadata['matched_queries'])[:150]}[/]")
        console.print(f"   {doc.page_content[:240].strip()}…\n")
    success(
        "Evidence ready for grading",
        {"Question": QUESTION.id, "Passages": len(evidence)},
        next_step="week5: the grader reads these passages next to the rubric",
    )
    tip("🧪 TRY: replace SAM_ANSWER with something off-topic. The evidence barely changes. That's the point.")


if __name__ == "__main__":
    try:
        main()
    except GradeWiseError as err:
        err.show()
        sys.exit(1)
