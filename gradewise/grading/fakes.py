"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 5: Fake models for free, offline runs                       │
│                                                                      │
│  What this file does                                                 │
│    Two stand-ins for a real chat model, so the tests (and you,       │
│    without an API key) can run the whole grading pipeline for free:  │
│                                                                      │
│      ScriptedChatModel   → replays the messages you give it, in      │
│                            order, tool calls included                │
│      KeywordGraderModel  → a deterministic "examiner" that reads the │
│                            rubric out of the prompt and awards marks │
│                            by keyword overlap. Dumb, but honest      │
│                                                                      │
│  ⚠️ These are NOT graders you should trust. They exist to exercise   │
│     the plumbing: tools, structured output, Send, interrupts.        │
│  ⏭ Used by: tests/test_grading.py, test_orchestrator.py, demos      │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def tool_call_message(name: str, args: dict[str, Any], call_id: str | None = None) -> AIMessage:
    """An AIMessage that calls one tool, the way a real model would."""
    call_id = call_id or f"call_{uuid.uuid4().hex[:8]}"
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def _tool_names(tools: Any) -> list[str]:
    names = []
    for t in tools or []:
        if isinstance(t, dict):
            names.append(t.get("name") or t.get("function", {}).get("name", ""))
        else:
            names.append(getattr(t, "name", None) or getattr(t, "__name__", str(t)))
    return names


class ScriptedChatModel(GenericFakeChatModel):
    """GenericFakeChatModel that accepts tools.

    💡 create_agent calls `model.bind_tools(...)` before every model call. The real models send the tool
       schemas to the provider; our fake just ignores them and replays its script, so we can script the
       exact tool calls (including the structured-output tool "GradeResult") we want to test.
    """

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ARG002
        return self


class KeywordGraderModel(BaseChatModel):
    """A deterministic offline examiner for the GradeResult agent.

    - First turn (if tools are bound): calls retrieve_reference with the first criterion as the query.
    - Then: reads the <rubric> JSON and <student_answer> from the prompt and calls the "GradeResult" tool.
      A criterion earns full marks if at least half its keywords appear, half marks if any appear.
    """

    retrieve_first: bool = True
    bound_tools: list[str] = []
    confidence: float = 0.85

    @property
    def _llm_type(self) -> str:
        return "keyword-grader-fake"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # noqa: ARG002
        return self.model_copy(update={"bound_tools": _tool_names(tools)})

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ARG002
        message = self._respond(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _respond(self, messages: list[BaseMessage]) -> AIMessage:
        prompt = next((m.content for m in messages if isinstance(m, HumanMessage) and "<rubric>" in str(m.content)), "")
        rubric = json.loads(_between(str(prompt), "<rubric>", "</rubric>") or "{}")
        answer = _between(str(prompt), "<student_answer>", "</student_answer>").lower()
        tool_results = [m for m in messages if isinstance(m, ToolMessage)]
        if self.retrieve_first and "retrieve_reference" in self.bound_tools and not tool_results:
            first = (rubric.get("criteria") or [{}])[0].get("description", "course notes")
            return tool_call_message("retrieve_reference", {"query": first[:120]})
        citations = re.findall(r'cite="([^"]+)"', " ".join(str(m.content) for m in tool_results))
        citations += re.findall(r'cite="([^"]+)"', str(prompt))
        criteria = []
        for c in rubric.get("criteria", []):
            keywords = [k.lower() for k in c.get("keywords", [])] or c["description"].lower().split()[:4]
            hits = [k for k in keywords if _fuzzy_in(k, answer)]
            share = len(hits) / len(keywords)
            awarded = c["marks"] if share >= 0.5 else (c["marks"] / 2 if hits else 0.0)
            criteria.append(
                {
                    "criterion_id": c["criterion_id"],
                    "awarded": awarded,
                    "justification": f"Mentions {', '.join(hits)}." if hits else "The key idea is not mentioned.",
                    "evidence": citations[:1] if hits else [],
                }
            )
        total = sum(c["awarded"] for c in criteria)
        best = max(criteria, key=lambda c: c["awarded"], default=None)
        worst = min(criteria, key=lambda c: c["awarded"], default=None)
        feedback = "Good start."
        if best and worst:
            feedback = best["justification"]
            if worst["awarded"] < next(
                c["marks"] for c in rubric["criteria"] if c["criterion_id"] == worst["criterion_id"]
            ):
                feedback += f" To improve, revisit {worst['criterion_id']}."
        args = {
            "question_id": rubric.get("question_id", "?"),
            "awarded_marks": total,
            "max_marks": rubric.get("max_marks", total or 1),
            "criteria": criteria,
            "feedback": feedback,
            "confidence": self.confidence,
            "needs_review": False,
        }
        return tool_call_message("GradeResult", args)


def _between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def _fuzzy_in(keyword: str, text: str) -> bool:
    """Keyword present, ignoring punctuation; multi-word keywords need every word (principle 3, roughly)."""
    words = re.findall(r"[a-z0-9]+", keyword)
    return bool(words) and all(w in text for w in words)
