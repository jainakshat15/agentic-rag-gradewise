"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧪 tests/test_models.py: MODEL_MODE wiring, checked OFFLINE         │
│                                                                      │
│  What: builds the real models for each mode and inspects exactly     │
│        what LangChain would send, with no key and no network:        │
│          • OpenAI: a fake HTTP transport captures the request body   │
│            and answers like the Responses API would.                 │
│          • Gemini: LangChain's own request builder (_prepare_request)│
│  💡 Live calls live in tests/test_live_smoke.py (opt-in).            │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from PIL import Image
from pydantic import BaseModel

from gradewise.models import (
    GOOGLE_MAX_RETRIES,
    RetryingEmbeddings,
    is_rate_limit,
    make_chat_model,
    retry_on_rate_limit,
)
from gradewise.transcription.pdf_to_images import image_block

LUNA = "openai:gpt-6-luna"  # custom mode, OpenAI directly
GEMINI = "google_genai:gemini-3.1-flash-lite"  # custom mode, Google directly


class Mark(BaseModel):
    mark: int
    reason: str


@tool
def lookup(topic: str) -> str:
    """Look up a course topic."""
    return topic


def _usage() -> dict:
    return {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


def _response(output: list[dict]) -> dict:
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-6-luna",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": _usage(),
    }


def _text(text: str) -> dict:
    return {
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


class FakeOpenAI:
    """An httpx transport that records each request body and replies with the next canned output."""

    def __init__(self, *outputs: list[dict]) -> None:
        self.outputs = list(outputs)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append({"path": request.url.path, "body": json.loads(request.content)})
        return httpx.Response(200, json=_response(self.outputs.pop(0)))

    def model(self, tier: str, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(self))
        return make_chat_model(LUNA, tier=tier, api_key="sk-offline", http_client=client, **kwargs)


@pytest.fixture(autouse=True)
def _offline_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline")
    monkeypatch.setenv("GOOGLE_API_KEY", "offline")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-offline")


# ── 🧠 gpt-6-luna called at OpenAI directly (custom mode) ─────────────────────────────────────────────
@pytest.mark.parametrize(("tier", "effort"), [("chat", "low"), ("smart", "medium"), ("vision", "low")])
def test_luna_tiers_use_responses_api_and_reasoning_effort(tier, effort):
    fake = FakeOpenAI([_text("hello")])
    llm = fake.model(tier, temperature=0)
    reply = llm.invoke("hi")
    body = fake.requests[0]["body"]
    assert fake.requests[0]["path"].endswith("/responses")  # 💡 the Responses API, not chat/completions
    assert body["model"] == "gpt-6-luna"
    assert body["reasoning"] == {"effort": effort}
    assert "temperature" not in body  # ⚠️ reasoning models reject it; we drop temperature=0
    assert reply.text == "hello"


def test_luna_tool_call_with_reasoning():
    call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": json.dumps({"topic": "deadlock"}),
        "status": "completed",
    }
    fake = FakeOpenAI([call])
    reply = fake.model("chat").bind_tools([lookup]).invoke("What is deadlock?")
    body = fake.requests[0]["body"]
    assert body["reasoning"] == {"effort": "low"}  # tools + reasoning: fine on the Responses API
    assert body["tools"][0]["name"] == "lookup"
    assert reply.tool_calls[0]["name"] == "lookup"
    assert reply.tool_calls[0]["args"] == {"topic": "deadlock"}


def test_luna_structured_output():
    fake = FakeOpenAI([_text(json.dumps({"mark": 2, "reason": "names both conditions"}))])
    result = fake.model("smart").with_structured_output(Mark).invoke("Mark this.")
    body = fake.requests[0]["body"]
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["name"] == "Mark"
    assert body["reasoning"] == {"effort": "medium"}
    assert result == Mark(mark=2, reason="names both conditions")


def test_luna_structured_output_include_raw_keeps_usage():
    fake = FakeOpenAI([_text(json.dumps({"mark": 1, "reason": "ok"}))])
    result = fake.model("vision").with_structured_output(Mark, include_raw=True).invoke("Mark this.")
    assert result["parsed"] == Mark(mark=1, reason="ok")
    assert result["raw"].usage_metadata["input_tokens"] == 10


def test_luna_image_input():
    fake = FakeOpenAI([_text("I see a page")])
    message = HumanMessage(content=[{"type": "text", "text": "Read this"}, image_block(Image.new("RGB", (8, 8)))])
    fake.model("vision").invoke([SystemMessage("Transcribe."), message])
    content = fake.requests[0]["body"]["input"][-1]["content"]
    image = next(part for part in content if part["type"] == "input_image")
    assert image["image_url"].startswith("data:image/jpeg;base64,")


def test_luna_agent_with_tools_and_response_format():
    """The grader's shape: create_agent(tools=..., response_format=...) on the SMART tier."""
    from langchain.agents import create_agent

    call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": json.dumps({"topic": "deadlock"}),
        "status": "completed",
    }
    fake = FakeOpenAI([call], [_text(json.dumps({"mark": 3, "reason": "complete"}))])
    agent = create_agent(fake.model("smart"), tools=[lookup], response_format=Mark)
    out = agent.invoke({"messages": [HumanMessage("Grade the answer.")]})
    assert out["structured_response"] == Mark(mark=3, reason="complete")
    first, second = (r["body"] for r in fake.requests)
    assert first["tools"][0]["name"] == "lookup" and first["reasoning"] == {"effort": "medium"}
    assert any(item.get("type") == "function_call_output" for item in second["input"])


def test_explicit_reasoning_effort_wins():
    fake = FakeOpenAI([_text("ok")])
    fake.model("smart", reasoning_effort="high").invoke("hi")
    assert fake.requests[0]["body"]["reasoning"] == {"effort": "high"}


# ── 🔷 Gemini called at Google directly (custom mode) ──────────────────────────────────────────────────
def _gemini(tier: str, **kwargs):
    return make_chat_model(GEMINI, tier=tier, **kwargs)


def test_gemini_retries_429s_with_more_attempts():
    llm = _gemini("chat")
    assert llm.max_retries == GOOGLE_MAX_RETRIES
    request = llm._prepare_request([HumanMessage("hi")])
    assert request["config"].http_options.retry_options.attempts == GOOGLE_MAX_RETRIES


def test_gemini_smart_tier_thinks_a_little_more():
    assert _gemini("smart").reasoning_effort == "low"
    assert _gemini("chat").reasoning_effort is None  # 💡 the model default ("minimal")


def test_gemini_tools_structured_output_and_images_build_requests():
    llm = _gemini("vision", temperature=0)
    messages = [
        SystemMessage("Transcribe."),
        HumanMessage(content=[{"type": "text", "text": "Read"}, image_block(Image.new("RGB", (8, 8)))]),
    ]
    request = llm._prepare_request(messages)
    parts = request["contents"][0].parts
    assert any(p.inline_data and p.inline_data.mime_type == "image/jpeg" for p in parts)

    bound = llm.bind_tools([lookup])
    request = llm._prepare_request(messages, **bound.kwargs)
    assert request["config"].tools[0].function_declarations[0].name == "lookup"

    structured = llm.with_structured_output(Mark)
    kwargs = {k: v for k, v in structured.first.kwargs.items() if k != "ls_structured_output_format"}
    request = llm._prepare_request(messages, **kwargs)
    assert request["config"].response_mime_type == "application/json"


# ── ⏳ Rate-limit retries ──────────────────────────────────────────────────
class ResourceExhausted(Exception):
    """Stands in for google.api_core.exceptions.ResourceExhausted."""


def test_is_rate_limit_recognises_429s():
    assert is_rate_limit(ResourceExhausted("quota"))
    assert is_rate_limit(RuntimeError("429 RESOURCE_EXHAUSTED: slow down"))
    assert not is_rate_limit(ValueError("bad schema"))


def test_retry_on_rate_limit_backs_off_then_succeeds(monkeypatch):
    waits = []
    monkeypatch.setattr("gradewise.models.time.sleep", waits.append)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ResourceExhausted("429")
        return "ok"

    assert retry_on_rate_limit(flaky, attempts=5) == "ok"
    assert len(waits) == 2 and waits[1] > waits[0]


def test_retry_does_not_hide_real_errors(monkeypatch):
    monkeypatch.setattr("gradewise.models.time.sleep", lambda _s: None)

    def broken():
        raise ValueError("not a rate limit")

    with pytest.raises(ValueError):
        retry_on_rate_limit(broken)


def test_retrying_embeddings_wraps_and_keeps_model_name(monkeypatch):
    from langchain_core.embeddings import DeterministicFakeEmbedding

    monkeypatch.setattr("gradewise.models.time.sleep", lambda _s: None)
    inner = DeterministicFakeEmbedding(size=8)
    fails = {"left": 1}
    original = inner.embed_documents

    def flaky(texts):
        if fails["left"]:
            fails["left"] -= 1
            raise ResourceExhausted("429")
        return original(texts)

    object.__setattr__(inner, "embed_documents", flaky)
    wrapped = RetryingEmbeddings(inner)
    assert len(wrapped.embed_documents(["a", "b"])) == 2
    assert len(wrapped.embed_query("a")) == 8
