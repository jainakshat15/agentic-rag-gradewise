"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 4 · Day 1: Agentic RAG, the demo                            │
│                                                                      │
│  What this file does                                                 │
│    Builds the agentic RAG graph from gradewise/rag/agentic_rag.py    │
│    and streams it on a question, so you can watch each node run:     │
│    route → retrieve → grade → (rewrite → retrieve …) → answer.       │
│                                                                      │
│  Run                                                                 │
│    uv run week4/project/agentic_rag.py "Why can't deadlock happen    │
│        if we break circular wait?"                                   │
│    uv run week4/project/agentic_rag.py --diagram   # Mermaid only    │
│                                                                      │
│  ⏮ Built on: week2 retrievers, week3 StateGraph                      │
│  ⏭ Used by: week5 grading (retrieval as a tool)                      │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage  # noqa: E402

from gradewise.console import banner, console, step, success  # noqa: E402
from gradewise.errors import GradeWiseError  # noqa: E402
from gradewise.rag import build_agentic_rag, unique_citations  # noqa: E402

DEFAULT_QUESTION = "Why does breaking the circular-wait condition prevent deadlock?"


def show_diagram() -> None:
    """Print the graph as Mermaid. 💡 No API calls: we hand it a placeholder model."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    graph = build_agentic_rag(llm=FakeListChatModel(responses=["-"]), retriever=_NoRetriever())
    console.print(graph.get_graph().draw_mermaid())
    console.print("[muted]Paste this into https://mermaid.live to see it drawn.[/]")


class _NoRetriever:
    def invoke(self, _query: str):
        return []


def main(question: str) -> None:
    banner("Week 4 · Agentic RAG", question)
    graph = build_agentic_rag()
    docs = []
    final = None
    # 💡 stream_mode="updates" yields {node_name: what_that_node_returned} after each step.
    for update in graph.stream({"messages": [HumanMessage(question)], "rewrites": 0}, stream_mode="updates"):
        for node, output in update.items():
            message = (output or {}).get("messages", [None])[-1]
            step(f"[primary]{node}[/]")
            if message is None:
                continue
            if getattr(message, "tool_calls", None):
                console.print(f"    [muted]→ search: {message.tool_calls[0]['args'].get('query')}[/]")
            elif getattr(message, "artifact", None):
                docs = message.artifact
                console.print(f"    [muted]→ {len(docs)} chunks retrieved[/]")
            else:
                final = message
                console.print(f"    [muted]→ {message.text[:200]}[/]")
    console.print(f"\n[primary]Answer:[/] {final.text if final else '(none)'}\n")
    success("Agentic RAG complete", {"Sources": "\n".join(unique_citations(docs)) or "none"})


if __name__ == "__main__":
    try:
        if "--diagram" in sys.argv:
            show_diagram()
        else:
            main(" ".join(a for a in sys.argv[1:] if not a.startswith("--")) or DEFAULT_QUESTION)
    except GradeWiseError as err:
        err.show()
        sys.exit(1)
