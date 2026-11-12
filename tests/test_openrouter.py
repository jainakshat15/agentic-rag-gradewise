"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧪 tests/test_openrouter.py: cheap + free modes, checked OFFLINE    │
│                                                                      │
│  What: builds the real OpenRouter models for MODEL_MODE=cheap and    │
│        free, swaps in a fake HTTP transport, and checks exactly what │
│        would be sent to https://openrouter.ai/api/v1 (and that the   │
│        replies parse). No key, no network, no cost.                  │
│  💡 Live calls live in tests/test_live_smoke.py (opt-in).            │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json

import httpx
import openrouter
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from PIL import Image
from pydantic import BaseModel

from gradewise import models
from gradewise.config import MODE_MODELS, settings
from gradewise.errors import FreeLimitReachedError
from gradewise.transcription.pdf_to_images import image_block

LUNA = MODE_MODELS["cheap"]["chat"]
FREE = MODE_MODELS["free"]["chat"]


class Mark(BaseModel):
    mark: int
    reason: str


@tool
def lookup(topic: str) -> str:
    """Look up a course topic."""
    return topic


def completion(content=None, tool_calls=None, model="openai/gpt-6-luna") -> dict:
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "system_fingerprint": None,
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def call(name: str, args: dict) -> list[dict]:
    return [{"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]


class FakeOpenRouter:
    """Records each request to OpenRouter and answers with the next canned response (or HTTP status)."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append({"url": str(request.url), "headers": request.headers, "body": json.loads(request.content)})
        reply = self.replies.pop(0)
        if isinstance(reply, int):
            body = {"error": {"code": reply, "message": "Rate limit exceeded: free-models-per-min"}}
            return httpx.Response(reply, json=body)
        return httpx.Response(200, json=reply)

    def model(self, model: str, tier: str, **kwargs):
        llm = models.make_chat_model(model, tier=tier, **kwargs)
        client = httpx.Client(transport=httpx.MockTransport(self))
        llm.client = openrouter.OpenRouter(api_key="sk-or-offline", client=client)
        return llm

    @property
    def body(self) -> dict:
        return self.requests[-1]["body"]


@pytest.fixture(autouse=True)
def _offline_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-offline")
    monkeypatch.setattr("gradewise.models.time.sleep", lambda _s: None)


# ── 💰 cheap: openrouter:openai/gpt-6-luna ─────────────────────────────────
def test_cheap_mode_uses_openrouter_with_one_key():
    from gradewise.config import Settings

    s = Settings(_env_file=None)
    assert s.models == MODE_MODELS["cheap"] and s.required_keys == ["OPENROUTER_API_KEY"]
    assert Settings(_env_file=None, model_mode="free").required_keys == ["OPENROUTER_API_KEY"]


@pytest.mark.parametrize(("tier", "effort"), [("chat", "low"), ("smart", "medium"), ("vision", "low")])
def test_luna_tiers_send_reasoning_effort_and_no_temperature(tier, effort):
    fake = FakeOpenRouter(completion("hello"))
    llm = fake.model(LUNA, tier, temperature=0)
    assert type(llm).__name__ == "GradeWiseChatOpenRouter"
    reply = llm.invoke("hi")
    assert fake.requests[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert fake.requests[0]["headers"]["authorization"] == "Bearer sk-or-offline"
    assert fake.body["model"] == "openai/gpt-6-luna"
    assert fake.body["reasoning"] == {"effort": effort}
    assert "temperature" not in fake.body  # ⚠️ OpenAI reasoning models reject it
    assert reply.text == "hello"
    assert models.answered_by(reply) == "openai/gpt-6-luna"
    assert reply.usage_metadata["input_tokens"] == 10


def test_explicit_reasoning_effort_wins():
    fake = FakeOpenRouter(completion("ok"))
    fake.model(LUNA, "chat", reasoning_effort="high").invoke("hi")
    assert fake.body["reasoning"] == {"effort": "high"}


def test_luna_tool_call():
    fake = FakeOpenRouter(completion(tool_calls=call("lookup", {"topic": "deadlock"})))
    reply = fake.model(LUNA, "smart").bind_tools([lookup]).invoke("What is deadlock?")
    assert fake.body["tools"][0]["function"]["name"] == "lookup"
    assert fake.body["reasoning"] == {"effort": "medium"}  # tools + reasoning together
    assert reply.tool_calls[0]["args"] == {"topic": "deadlock"}


def test_luna_structured_output():
    fake = FakeOpenRouter(completion(tool_calls=call("Mark", {"mark": 2, "reason": "both conditions"})))
    result = fake.model(LUNA, "smart").with_structured_output(Mark).invoke("Mark this.")
    assert fake.body["tool_choice"]["function"]["name"] == "Mark"  # 💡 ChatOpenRouter's default: function_calling
    assert result == Mark(mark=2, reason="both conditions")


def test_luna_structured_output_include_raw_keeps_usage_and_model():
    fake = FakeOpenRouter(completion(tool_calls=call("Mark", {"mark": 1, "reason": "ok"})))
    result = fake.model(LUNA, "vision").with_structured_output(Mark, include_raw=True).invoke("Mark this.")
    assert result["parsed"] == Mark(mark=1, reason="ok")
    assert result["raw"].usage_metadata["output_tokens"] == 5
    assert models.answered_by(result["raw"]) == "openai/gpt-6-luna"


def test_image_blocks_become_data_urls():
    fake = FakeOpenRouter(completion("I see a page"))
    message = HumanMessage(content=[{"type": "text", "text": "Read"}, image_block(Image.new("RGB", (8, 8)))])
    fake.model(LUNA, "vision").invoke([SystemMessage("Transcribe."), message])
    parts = fake.body["messages"][-1]["content"]
    image = next(p for p in parts if p["type"] == "image_url")
    assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_agent_with_tools_and_response_format():
    """The grader's shape: create_agent(tools=..., response_format=...) on the SMART tier."""
    from langchain.agents import create_agent

    fake = FakeOpenRouter(
        completion(tool_calls=call("lookup", {"topic": "deadlock"})),
        completion(json.dumps({"mark": 3, "reason": "complete"})),  # 💡 native structured output: JSON text
    )
    agent = create_agent(fake.model(LUNA, "smart"), tools=[lookup], response_format=Mark)
    out = agent.invoke({"messages": [HumanMessage("Grade the answer.")]})
    assert out["structured_response"] == Mark(mark=3, reason="complete")
    first, second = (r["body"] for r in fake.requests)
    assert first["tools"][0]["function"]["name"] == "lookup"
    assert first["response_format"]["type"] == "json_schema" and first["reasoning"] == {"effort": "medium"}
    assert any(m["role"] == "tool" for m in second["messages"])


# ── 🆓 free: openrouter:openrouter/free ────────────────────────────────────
def test_free_router_keeps_temperature_and_reports_who_answered():
    fake = FakeOpenRouter(completion("hi", model="some-lab/some-model:free"))
    reply = fake.model(FREE, "chat", temperature=0).invoke("hi")
    assert fake.body["model"] == "openrouter/free"
    assert fake.body["temperature"] == 0 and "reasoning" not in fake.body
    assert models.answered_by(reply) == "some-lab/some-model:free"


def test_free_router_retries_a_structured_output_that_does_not_parse():
    fake = FakeOpenRouter(
        completion("sorry, here is prose instead of a tool call", model="a:free"),
        completion(tool_calls=call("Mark", {"mark": 1, "reason": "ok"}), model="b:free"),
    )
    result = fake.model(FREE, "smart").with_structured_output(Mark).invoke("Mark this.")
    assert result == Mark(mark=1, reason="ok") and len(fake.requests) == 2


def test_429_is_retried_with_backoff():
    fake = FakeOpenRouter(429, 429, completion("finally"))
    assert fake.model(FREE, "chat").invoke("hi").text == "finally"
    assert len(fake.requests) == 3


def test_daily_limit_is_a_friendly_error_not_a_retry_loop():
    fake = FakeOpenRouter(429)

    def daily(request):
        fake.requests.append(request)
        body = {"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-day"}}
        return httpx.Response(429, json=body)

    llm = models.make_chat_model(FREE, tier="chat")
    llm.client = openrouter.OpenRouter(api_key="x", client=httpx.Client(transport=httpx.MockTransport(daily)))
    with pytest.raises(FreeLimitReachedError) as exc:
        llm.invoke("hi")
    assert "DATA_SIZE=small" in str(exc.value) and len(fake.requests) == 1


def test_free_mode_warns_once_when_data_is_full(monkeypatch, capsys):
    monkeypatch.setattr(settings, "model_mode", "free")
    monkeypatch.setattr(models, "_warned_full_data", False)
    models.chat_model()
    models.chat_model()
    out = capsys.readouterr().out
    assert out.count("DATA_SIZE=small") == 1
    monkeypatch.setattr(settings, "data_size", "small")
    monkeypatch.setattr(models, "_warned_full_data", False)
    models.chat_model()
    assert "DATA_SIZE=small" not in capsys.readouterr().out


def test_free_banner_says_the_router_picks():
    from gradewise.config import Settings

    line = Settings(_env_file=None, model_mode="free").describe_models()
    assert "openrouter/free" in line and "router picks" in line and "$0" in line


# ── 🔢 Embeddings through OpenRouter ───────────────────────────────────────
@pytest.mark.parametrize("mode", ["cheap", "free"])
def test_embeddings_go_to_openrouter(monkeypatch, mode):
    monkeypatch.setattr(settings, "model_mode", mode)
    seen = []

    def handler(request):
        seen.append({"url": str(request.url), "body": json.loads(request.content)})
        data = [{"object": "embedding", "index": i, "embedding": [0.1, 0.2, 0.3]} for i in range(2)]
        return httpx.Response(200, json={"object": "list", "data": data, "model": "x", "usage": {}})

    emb = models.embeddings(http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert isinstance(emb, models.RetryingEmbeddings)
    assert emb.embed_documents(["deadlock", "mutex"]) == [[0.1, 0.2, 0.3]] * 2
    name = MODE_MODELS[mode]["embedding"].split(":", 1)[1]
    assert seen[0]["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert seen[0]["body"]["model"] == name
    assert seen[0]["body"]["input"] == ["deadlock", "mutex"]  # 💡 raw text, not token arrays
    assert seen[0]["body"]["encoding_format"] == "float"


def test_store_keys_for_openrouter_embeddings():
    from gradewise.ingestion import embedding_key

    for mode in ("cheap", "free"):
        model = MODE_MODELS[mode]["embedding"]
        emb = models.openrouter_embeddings(model.split(":", 1)[1])
        assert embedding_key(emb) == embedding_key(model=model)
    assert embedding_key(model=MODE_MODELS["free"]["embedding"]) == "openrouter_nvidia_nemotron-3.5-embed-1b_free"


def test_prices():
    from gradewise import pricing

    assert pricing.chat_price(LUNA).input == 0.10 and pricing.chat_price(LUNA).output == 0.50
    assert pricing.chat_price("openrouter:openai/gpt-6.1-terra").output == 5.0
    assert pricing.embedding_price(MODE_MODELS["cheap"]["embedding"]) == 0.02
    assert pricing.chat_cost(10**6, 10**6, FREE) == 0.0
    assert pricing.embedding_price(MODE_MODELS["free"]["embedding"]) == 0.0
    assert pricing.format_cost(0.0, FREE) == "$0 (free tier)"


def test_no_credits_is_a_friendly_error():
    from gradewise.errors import NoCreditsError

    def broke(request):
        return httpx.Response(402, json={"error": {"code": 402, "message": "Insufficient credits"}})

    llm = models.make_chat_model(LUNA, tier="chat")
    llm.client = openrouter.OpenRouter(api_key="x", client=httpx.Client(transport=httpx.MockTransport(broke)))
    with pytest.raises(NoCreditsError) as exc:
        llm.invoke("hi")
    assert "openrouter.ai/settings/credits" in str(exc.value)
