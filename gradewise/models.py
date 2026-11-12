"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 1 · Day 2: One place to create models                       │
│                                                                      │
│  What this file does                                                 │
│    Wraps LangChain's `init_chat_model` so every notebook gets its    │
│    models the same way, with a clear error if a key is missing.      │
│                                                                      │
│      chat_model()     → everyday model (CHAT tier)                   │
│      grading_model()  → the careful marker (SMART tier, week 5)      │
│      vision_model()   → reads handwriting (VISION tier, week 4)      │
│      embeddings()     → turns text into vectors (week 2)             │
│                                                                      │
│  💰 MODEL_MODE (in .env) picks the models, all through OpenRouter    │
│     with ONE key (OPENROUTER_API_KEY):                               │
│       cheap → openai/gpt-6-luna      free → openrouter/free ($0)     │
│       custom → your *_MODEL variables (any provider)                 │
│                                                                      │
│  🧠 gpt-6-luna is a reasoning model. The tiers differ only in        │
│     reasoning effort: SMART thinks harder (medium) than CHAT and     │
│     VISION (low).                                                    │
│                                                                      │
│  🔁 RECAP: model names come from gradewise/config.py (your .env)     │
│  ⏭ Used by: every notebook from week1/day2.ipynb onwards            │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings

from gradewise.config import FREE_ROUTER, OPENROUTER_BASE_URL, PROVIDER_KEYS, settings
from gradewise.errors import FreeLimitReachedError, InvalidModelError, MissingAPIKeyError, NoCreditsError

# 💡 Kept under its old name: notebooks and tests import it.
_PROVIDER_KEYS = PROVIDER_KEYS

# 🧠 Reasoning effort per tier and provider. Never applied if you pass reasoning_effort / reasoning yourself.
#    ⚠️ gpt-6-luna on OpenAI's Chat Completions API only calls tools with reasoning_effort="none"; the
#       Responses API allows tools WITH reasoning. Through OpenRouter we send `reasoning: {"effort": ...}`
#       and OpenRouter talks to OpenAI for us. Calling OpenAI directly (MODEL_MODE=custom, "openai:...")
#       we switch on the Responses API ourselves.
TIER_REASONING: dict[str, dict[str, str]] = {
    "openrouter": {"chat": "low", "smart": "medium", "vision": "low"},  # only for openai/ reasoning models
    "openai": {"chat": "low", "smart": "medium", "vision": "low"},
    "google_genai": {"smart": "low"},  # 💡 Gemini's default is "minimal": fast, and kind to rate limits
}

# 🧠 OpenAI reasoning families (gpt-5…, gpt-6…, o-series). They reject `temperature`.
OPENAI_REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")

# ⏳ Rate limits: a 429 means "slow down". We retry with exponential backoff + jitter.
RATE_LIMIT_RETRIES = 6
EMBEDDING_RETRIES = RATE_LIMIT_RETRIES
GOOGLE_MAX_RETRIES = 8  # custom mode with google_genai: the Google SDK's own backoff
# 🎲 openrouter/free picks a different model each call, so a structured-output reply occasionally
#    doesn't parse. We simply ask again (up to this many extra tries).
FREE_STRUCTURED_RETRIES = 2

_warned_full_data = False


def _split(model: str) -> tuple[str, str]:
    if ":" not in model:
        raise InvalidModelError(f'"{model}" has no provider prefix.')
    provider, name = model.split(":", 1)
    return provider, name


def _check(model: str) -> str:
    """Validate a "provider:model" string and make sure its API key is available."""
    provider, _name = _split(model)
    env_var = _PROVIDER_KEYS.get(provider, "")
    if env_var:
        # ⚠️ LangChain reads keys from the environment, so we copy them out of .env.
        secret = getattr(settings, env_var.lower(), None)
        if secret is not None and secret.get_secret_value() and not os.getenv(env_var):
            os.environ[env_var] = secret.get_secret_value()
        if not os.getenv(env_var):
            raise MissingAPIKeyError(
                f"{env_var} is not set, but your model is {model} (MODEL_MODE={settings.model_mode}).",
                fix=_key_fix(env_var),
            )
    return model


def _key_fix(env_var: str) -> str:
    if env_var == "OPENROUTER_API_KEY":
        return (
            "Create a key at https://openrouter.ai/keys, paste it after OPENROUTER_API_KEY= in .env and SAVE. "
            "MODEL_MODE=cheap needs a few dollars of OpenRouter credits; MODEL_MODE=free needs none. "
            "Then restart the kernel."
        )
    return (
        f"MODEL_MODE=custom uses this provider directly: paste your key after {env_var}= in .env and SAVE, "
        "or go back to MODEL_MODE=cheap / free (one OPENROUTER_API_KEY). Then restart the kernel."
    )


def _is_openai_reasoning(provider: str, name: str) -> bool:
    name = name.lower()
    if provider == "openrouter":
        return name.startswith("openai/") and name.removeprefix("openai/").startswith(OPENAI_REASONING_PREFIXES)
    return provider == "openai" and name.startswith(OPENAI_REASONING_PREFIXES)


def chat_model_kwargs(model: str, tier: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """The keyword arguments `make_chat_model` uses to build the model (no network, no key needed).

    💡 Handy for seeing what a tier does: chat_model_kwargs("openrouter:openai/gpt-6-luna", "smart").
    """
    provider, name = _split(model)
    kwargs = dict(kwargs)
    if provider == "openrouter":
        # 💡 ChatOpenRouter takes reasoning as a dict: {"effort": "medium"}.
        if "reasoning_effort" in kwargs:
            kwargs["reasoning"] = {"effort": kwargs.pop("reasoning_effort")}
        if _is_openai_reasoning(provider, name):
            effort = TIER_REASONING["openrouter"].get(tier or "")
            if effort and "reasoning" not in kwargs:
                kwargs["reasoning"] = {"effort": effort}
            # ⚠️ OpenAI reasoning models reject `temperature`; the course passes temperature=0 for
            #    repeatable extraction, so we drop it. (openrouter/free accepts it, so it stays there.)
            kwargs.pop("temperature", None)
    if provider == "openai" and _is_openai_reasoning(provider, name):
        # 🧠 Direct OpenAI (custom mode): the Responses API runs tools + structured output WITH reasoning.
        kwargs.setdefault("use_responses_api", True)
    if provider == "google_genai":
        kwargs.setdefault("max_retries", GOOGLE_MAX_RETRIES)
    return kwargs


def _tune(llm: Any, provider: str, tier: str | None, user_kwargs: dict[str, Any]) -> Any:
    """Apply the tier's reasoning effort and drop settings the model rejects (uses LangChain's model profile)."""
    if provider == "openrouter":
        return llm  # 💡 already handled in chat_model_kwargs
    profile = getattr(llm, "profile", None) or {}
    wants_effort = "reasoning_effort" not in user_kwargs and "reasoning" not in user_kwargs
    effort = TIER_REASONING.get(provider, {}).get(tier or "")
    levels = profile.get("reasoning_effort_levels") or []
    if wants_effort and effort and effort in levels and hasattr(llm, "reasoning_effort"):
        llm.reasoning_effort = effort
    # ⚠️ Reasoning models like gpt-6-luna reject `temperature` (profile: temperature=False).
    if profile.get("temperature") is False and getattr(llm, "temperature", None) is not None:
        llm.temperature = None
    return llm


def _warn_free_full_data() -> None:
    """One line, once per session: free mode has a daily request cap, and full data makes hundreds of calls."""
    global _warned_full_data
    from gradewise import datasets

    if settings.is_free and not datasets.is_small() and not _warned_full_data:
        _warned_full_data = True
        from gradewise.console import warn

        warn(
            "MODEL_MODE=free with DATA_SIZE=full: a full class run makes hundreds of calls, and free models "
            "allow about 50 requests a day. Set DATA_SIZE=small in .env."
        )


def make_chat_model(model: str, *, tier: str | None = None, **kwargs: Any):
    """Create any chat model from a "provider:model" string.

    tier: "chat", "smart" or "vision" (sets the reasoning effort), or None to leave the model's default.
    """
    provider, name = _split(_check(model))
    if provider == "openrouter":
        if settings.is_free:
            _warn_free_full_data()
        params = chat_model_kwargs(model, tier, **kwargs)
        if name == FREE_ROUTER or name.endswith(":free"):
            params.setdefault("structured_retries", FREE_STRUCTURED_RETRIES)
        return openrouter_chat_class()(model=name, **params)
    from langchain.chat_models import init_chat_model

    llm = init_chat_model(model, **chat_model_kwargs(model, tier, **kwargs))
    return _tune(llm, provider, tier, kwargs)


def chat_model(**kwargs: Any):
    """The everyday model (CHAT tier: gpt-6-luna at low effort in cheap mode)."""
    return make_chat_model(settings.chat_model, tier="chat", **kwargs)


def grading_model(**kwargs: Any):
    """The careful marking model (SMART tier: gpt-6-luna at medium effort in cheap mode)."""
    return make_chat_model(settings.grading_model, tier="smart", **kwargs)


def vision_model(**kwargs: Any):
    """The handwriting model (VISION tier: must support images)."""
    return make_chat_model(settings.transcription_model, tier="vision", **kwargs)


# ── ⏳ Rate limits ─────────────────────────────────────────────────────────
def is_rate_limit(exc: BaseException) -> bool:
    """True for "slow down" errors: HTTP 429 (OpenRouter's TooManyRequests, OpenAI's RateLimitError, …)."""
    names = {type(e).__name__ for e in (exc, exc.__cause__, exc.__context__) if e is not None}
    if names & {
        "ResourceExhausted",
        "RateLimitError",
        "GoogleRateLimitError",
        "TooManyRequests",
        "TooManyRequestsResponseError",
    }:
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = str(exc)
    return code == 429 or "429" in text or "RESOURCE_EXHAUSTED" in text


def is_daily_limit(exc: BaseException) -> bool:
    """True when the free models' DAILY cap is used up: waiting a minute won't help, so don't retry."""
    text = str(exc).lower()
    return is_rate_limit(exc) and ("per-day" in text or "per day" in text)


def _backoff(attempt: int, base_delay: float) -> float:
    return min(60.0, base_delay * 2 ** (attempt - 1)) * (0.75 + random.random() / 2)


def _daily_limit_error(exc: BaseException) -> FreeLimitReachedError:
    return FreeLimitReachedError(f"The free models' daily request limit is used up ({type(exc).__name__}).")


def retry_on_rate_limit(fn, *args: Any, attempts: int = RATE_LIMIT_RETRIES, base_delay: float = 2.0, **kwargs: Any):
    """Call fn(*args, **kwargs), retrying 429s with exponential backoff + jitter (2s, 4s, 8s … max 60s)."""
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if getattr(exc, "status_code", None) == 402:
                raise NoCreditsError(f"OpenRouter says payment required ({type(exc).__name__}).") from exc
            if is_daily_limit(exc):
                raise _daily_limit_error(exc) from exc
            if attempt == attempts or not is_rate_limit(exc):
                raise
            delay = _backoff(attempt, base_delay)
            from gradewise.console import tip

            tip(f"Rate limited (429): waiting {delay:.0f}s, then retry {attempt}/{attempts - 1}…")
            time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


async def aretry_on_rate_limit(fn, *args: Any, attempts: int = RATE_LIMIT_RETRIES, base_delay: float = 2.0, **kw):
    """The async twin of retry_on_rate_limit."""
    for attempt in range(1, attempts + 1):
        try:
            return await fn(*args, **kw)
        except Exception as exc:
            if getattr(exc, "status_code", None) == 402:
                raise NoCreditsError(f"OpenRouter says payment required ({type(exc).__name__}).") from exc
            if is_daily_limit(exc):
                raise _daily_limit_error(exc) from exc
            if attempt == attempts or not is_rate_limit(exc):
                raise
            await asyncio.sleep(_backoff(attempt, base_delay))
    raise RuntimeError("unreachable")  # pragma: no cover


# ── 🔀 OpenRouter ──────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def openrouter_chat_class():
    """`ChatOpenRouter` plus three GradeWise extras (built on first use, so importing models.py stays light):

    1. 429s are retried with backoff (the OpenRouter SDK itself only retries 5xx errors);
    2. every reply's `response_metadata["model_name"]` says which model REALLY answered
       (with openrouter/free, the router picks one per call);
    3. with the free router, `with_structured_output` asks again if a reply doesn't parse.
    """
    from langchain_core.exceptions import OutputParserException
    from langchain_core.runnables import RunnableLambda
    from langchain_openrouter import ChatOpenRouter
    from pydantic import ValidationError

    def _require_output(value):
        if value is None:
            raise OutputParserException("The model replied without the structured output.")
        return value

    class GradeWiseChatOpenRouter(ChatOpenRouter):
        rate_limit_retries: int = RATE_LIMIT_RETRIES
        structured_retries: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            parent = super()._generate
            return retry_on_rate_limit(
                parent, messages, stop=stop, run_manager=run_manager, attempts=self.rate_limit_retries, **kwargs
            )

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            parent = super()._agenerate
            return await aretry_on_rate_limit(
                parent, messages, stop=stop, run_manager=run_manager, attempts=self.rate_limit_retries, **kwargs
            )

        def _create_chat_result(self, response):
            result = super()._create_chat_result(response)
            answered_by = (result.llm_output or {}).get("model_name")
            for generation in result.generations:
                if answered_by:
                    generation.message.response_metadata.setdefault("model_name", answered_by)
            return result

        def with_structured_output(self, schema=None, *, include_raw: bool = False, **kwargs):
            runnable = super().with_structured_output(schema, include_raw=include_raw, **kwargs)
            if self.structured_retries and not include_raw:
                # 💡 A reply with no tool call parses to None: treat that as a failure too, and ask again.
                return (runnable | RunnableLambda(_require_output)).with_retry(
                    retry_if_exception_type=(OutputParserException, ValidationError, ValueError),
                    stop_after_attempt=1 + self.structured_retries,
                    wait_exponential_jitter=False,
                )
            return runnable

    return GradeWiseChatOpenRouter


def answered_by(message: Any) -> str | None:
    """Which model actually answered: `response_metadata["model_name"]` (or None)."""
    meta = getattr(message, "response_metadata", None) or {}
    return meta.get("model_name") or meta.get("model")


# ── 🔢 Embeddings ──────────────────────────────────────────────────────────
class RetryingEmbeddings(Embeddings):
    """Wraps any LangChain Embeddings and retries rate-limit errors."""

    def __init__(self, inner: Embeddings, *, attempts: int = EMBEDDING_RETRIES) -> None:
        self.inner = inner
        self.attempts = attempts

    @property
    def model(self) -> str | None:  # 💡 so the vector store can key itself by model name
        return getattr(self.inner, "model", None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return retry_on_rate_limit(self.inner.embed_documents, texts, attempts=self.attempts)

    def embed_query(self, text: str) -> list[float]:
        return retry_on_rate_limit(self.inner.embed_query, text, attempts=self.attempts)

    def __repr__(self) -> str:
        return f"RetryingEmbeddings({self.inner!r})"


def openrouter_embeddings(name: str, **kwargs: Any) -> Embeddings:
    """OpenRouter embeddings. LangChain has no OpenRouter embeddings class, but OpenRouter's
    /api/v1/embeddings speaks OpenAI's format, so OpenAIEmbeddings works with OpenRouter's base URL.

    ⚠️ check_embedding_ctx_length=False sends raw text (OpenRouter doesn't accept OpenAI's token arrays),
       and encoding_format="float" avoids base64 replies some providers can't produce.
    """
    from langchain_openai import OpenAIEmbeddings

    model_kwargs = {"encoding_format": "float", **kwargs.pop("model_kwargs", {})}
    params = {
        "model": name,
        "base_url": OPENROUTER_BASE_URL,
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "check_embedding_ctx_length": False,
        "model_kwargs": model_kwargs,
        **kwargs,
    }
    return RetryingEmbeddings(OpenAIEmbeddings(**params))


def embeddings(**kwargs: Any):
    """The embedding model: openai/text-embedding-3-small (cheap) or nvidia/nemotron-3.5-embed-1b:free (free)."""
    model = _check(settings.embedding_model)
    provider, name = _split(model)
    if provider == "openrouter":
        return openrouter_embeddings(name, **kwargs)
    from langchain.embeddings import init_embeddings

    emb = init_embeddings(model, **kwargs)
    return RetryingEmbeddings(emb) if provider == "google_genai" else emb
