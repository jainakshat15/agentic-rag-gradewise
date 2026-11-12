"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🔴 tests/test_live_smoke.py: LIVE OpenRouter calls (opt-in)         │
│                                                                      │
│  ⚠️ These tests call the REAL OpenRouter API. They are SKIPPED       │
│     unless BOTH are true:                                            │
│       1. OPENROUTER_API_KEY is set (in .env or the environment);     │
│       2. you opt in with GRADEWISE_LIVE_TESTS=1.                     │
│                                                                      │
│  Run (PowerShell):  $env:GRADEWISE_LIVE_TESTS=1; uv run pytest -m live│
│  Run (Mac/Linux):   GRADEWISE_LIVE_TESTS=1 uv run pytest -m live     │
│                                                                      │
│  Each mode must: chat · structured output · call a tool (CHAT and    │
│  SMART tiers) · read an image · embed text.                          │
│  💰 cheap: a fraction of a cent. free: $0, but ~12 of your 50 free   │
│     requests for the day.                                            │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from PIL import Image, ImageDraw
from pydantic import BaseModel

from gradewise.config import MODE_MODELS, settings
from gradewise.models import _check, answered_by, make_chat_model, openrouter_embeddings
from gradewise.transcription.pdf_to_images import image_block


def _has_key() -> bool:
    secret = settings.openrouter_api_key
    return bool((secret and secret.get_secret_value()) or os.getenv("OPENROUTER_API_KEY"))


LIVE = pytest.mark.skipif(
    os.getenv("GRADEWISE_LIVE_TESTS") != "1" or not _has_key(),
    reason="live test: set GRADEWISE_LIVE_TESTS=1 and OPENROUTER_API_KEY to run",
)
MODES = [pytest.param(m, marks=[pytest.mark.live, LIVE], id=m) for m in ("cheap", "free")]


class Capital(BaseModel):
    country: str
    capital: str


@tool
def add(a: int, b: int) -> int:
    """Add two whole numbers."""
    return a + b


def _model(mode: str, tier: str):
    return make_chat_model(MODE_MODELS[mode][tier], tier=tier)


@pytest.mark.parametrize("mode", MODES)
def test_live_chat(mode):
    reply = _model(mode, "chat").invoke("Reply with exactly one word: pong")
    assert "pong" in reply.text.lower()
    print(f"{mode}: answered by {answered_by(reply)}")  # 💡 with openrouter/free, the router's pick
    assert answered_by(reply)


@pytest.mark.parametrize("mode", MODES)
def test_live_structured_output(mode):
    result = _model(mode, "smart").with_structured_output(Capital).invoke("What is the capital of France?")
    assert isinstance(result, Capital) and "paris" in result.capital.lower()


@pytest.mark.parametrize("tier", ["chat", "smart"])
@pytest.mark.parametrize("mode", MODES)
def test_live_tool_call(mode, tier):
    """⚠️ The important one for gpt-6-luna: tools WITH reasoning effort low (chat) and medium (smart)."""
    reply = _model(mode, tier).bind_tools([add], tool_choice="any").invoke("Use the add tool: what is 19 + 23?")
    assert reply.tool_calls and reply.tool_calls[0]["name"] == "add"
    assert reply.tool_calls[0]["args"] == {"a": 19, "b": 23}


@pytest.mark.parametrize("mode", MODES)
def test_live_temperature_is_accepted(mode):
    """The course passes temperature=0 for extraction: it must not break either mode."""
    reply = make_chat_model(MODE_MODELS[mode]["chat"], tier="chat", temperature=0).invoke("Say OK")
    assert reply.text


@pytest.mark.parametrize("mode", MODES)
def test_live_image(mode):
    img = Image.new("RGB", (360, 120), "white")
    ImageDraw.Draw(img).text((20, 40), "HELLO 42", fill="black", font_size=48)
    message = HumanMessage(
        content=[{"type": "text", "text": "What text is in this image? Reply with the text only."}, image_block(img)]
    )
    reply = _model(mode, "vision").invoke([message])
    assert "42" in reply.text


@pytest.mark.parametrize("mode", MODES)
def test_live_embedding(mode):
    model = _check(MODE_MODELS[mode]["embedding"])
    emb = openrouter_embeddings(model.split(":", 1)[1])
    vectors = emb.embed_documents(["deadlock", "mutual exclusion"])
    assert len(vectors) == 2 and len(vectors[0]) >= 256
    assert len(emb.embed_query("deadlock")) == len(vectors[0])
