"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                            │
│  💰 gradewise/pricing.py: what a call costs, roughly                  │
│                                                                      │
│  What this file does                                                 │
│    One table of list prices (US$ per 1M tokens) for the cost         │
│    readouts GradeWise prints. Nothing is billed from here.           │
│    OpenRouter passes the provider's price through:                   │
│                                                                      │
│      openai/gpt-6-luna      $0.10 in · $0.01 cached · $0.50 out      │
│      openai/gpt-6.1-terra   $1 in · $5 out (optional upgrade)        │
│      openai/text-embedding-3-small  $0.02                            │
│      openrouter/free, any ":free"   $0                               │
│                                                                      │
│    Prices verified 2026-11-12 (openrouter.ai/api/v1/models).         │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from dataclasses import dataclass

from gradewise.config import get_settings

PRICES_VERIFIED = "2026-11-12"


@dataclass(frozen=True)
class ChatPrice:
    """US$ per million tokens."""

    input: float
    output: float
    cached_input: float | None = None


CHAT_PRICES: dict[str, ChatPrice] = {
    "gpt-6-luna": ChatPrice(0.10, 0.50, 0.01),
    "gpt-6.1-terra": ChatPrice(1.00, 5.00),
    "gemini-3.1-flash-lite": ChatPrice(0.25, 1.50),  # custom mode, Google directly (paid tier)
}
EMBEDDING_PRICES: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "gemini-embedding-001": 0.15,
}
FREE = ChatPrice(0.0, 0.0, 0.0)
DEFAULT_CHAT = "gpt-6-luna"  # 💡 unknown models are estimated at the default model's price
DEFAULT_EMBEDDING = "text-embedding-3-small"


def _name(model: str) -> str:
    """'openrouter:openai/gpt-6-luna' → 'gpt-6-luna' (provider and vendor prefixes dropped)."""
    return model.split(":", 1)[-1].rsplit("/", 1)[-1]


def is_free_tier(model: str) -> bool:
    """True for $0 models: OpenRouter's free router (openrouter/free), any ":free" model, or local Ollama."""
    if ":" not in model:
        return False
    provider, name = model.split(":", 1)
    return provider == "ollama" or (provider == "openrouter" and (name == "openrouter/free" or name.endswith(":free")))


def chat_price(model: str | None = None) -> ChatPrice:
    """The price of a chat model (default: your CHAT model). $0 on the free tier."""
    model = model or get_settings().chat_model
    if is_free_tier(model):
        return FREE
    return CHAT_PRICES.get(_name(model), CHAT_PRICES[DEFAULT_CHAT])


def embedding_price(model: str | None = None) -> float:
    """US$ per million tokens for an embedding model (default: your EMBEDDING model). $0 on the free tier."""
    model = model or get_settings().embedding_model
    if is_free_tier(model):
        return 0.0
    return EMBEDDING_PRICES.get(_name(model), EMBEDDING_PRICES[DEFAULT_EMBEDDING])


def chat_cost(input_tokens: int, output_tokens: int, model: str | None = None, cached_tokens: int = 0) -> float:
    """Rough US$ for one or more calls. `cached_tokens` (part of input_tokens) are billed at the cached rate."""
    price = chat_price(model)
    cached = min(cached_tokens, input_tokens)
    cached_rate = price.cached_input if price.cached_input is not None else price.input
    return ((input_tokens - cached) * price.input + cached * cached_rate + output_tokens * price.output) / 1e6


def embedding_cost(tokens: int, model: str | None = None) -> float:
    """Rough US$ to embed `tokens` tokens."""
    return tokens / 1e6 * embedding_price(model)


def format_cost(usd: float, model: str | None = None) -> str:
    """'$0.00012', or '$0 (free tier)' when the model is on the free tier."""
    if model and is_free_tier(model):
        return "$0 (free tier)"
    return f"${usd:.5f}"


__all__ = [
    "CHAT_PRICES",
    "EMBEDDING_PRICES",
    "PRICES_VERIFIED",
    "ChatPrice",
    "chat_cost",
    "chat_price",
    "embedding_cost",
    "embedding_price",
    "format_cost",
    "is_free_tier",
]
