"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🔌 Week 6 · Day 2: The dashboard's only way to talk to GradeWise    │
│                                                                      │
│  What this file does                                                 │
│    A thin httpx wrapper around the FastAPI backend. The UI never     │
│    imports LangGraph or touches the database: it speaks HTTP, so it  │
│    can run in its own container, on its own machine, or be replaced  │
│    by a React app tomorrow without changing the backend.             │
│                                                                      │
│      get / post        → JSON in, JSON out, friendly ApiError        │
│      stream_events()   → reads the Server-Sent Events of a batch     │
│                                                                      │
│  Env : API_URL (default http://localhost:8000)                       │
│  ⏭ Used by: every page in gradewise/ui/pages/                       │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = httpx.Timeout(300.0, connect=5.0)  # 💡 ingest and real exam extraction can take a while


class ApiError(Exception):
    """An error from the API (or from failing to reach it), with GradeWise's four friendly fields."""

    def __init__(self, payload: dict[str, Any], status: int | None = None) -> None:
        self.payload = payload
        self.status = status
        super().__init__(payload.get("what", "Unknown error"))

    @property
    def title(self) -> str:
        return self.payload.get("error", "Something went wrong")


def _unreachable(exc: Exception) -> ApiError:
    return ApiError(
        {
            "error": "Can't reach the GradeWise API",
            "what": f"Nothing answered at {API_URL} ({type(exc).__name__}).",
            "cause": "The API isn't running, or API_URL points somewhere else.",
            "fix": "Start it in another terminal: uv run python -m gradewise.api --demo",
            "help": "week6/project/README.md",
        }
    )


def _raise_for(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if "what" not in payload:  # 💡 e.g. FastAPI's own 422 validation errors
        payload = {"error": f"HTTP {response.status_code}", "what": str(payload.get("detail", response.text))[:500]}
    raise ApiError(payload, response.status_code)


def request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except httpx.TransportError as exc:
        raise _unreachable(exc) from exc
    _raise_for(response)
    return response


def get(path: str, **params: Any) -> Any:
    clean = {k: v for k, v in params.items() if v is not None}
    return request("GET", path, params=clean).json()


def get_bytes(path: str, **params: Any) -> bytes:
    return request("GET", path, params={k: v for k, v in params.items() if v is not None}).content


def get_text(path: str, **params: Any) -> str:
    return request("GET", path, params={k: v for k, v in params.items() if v is not None}).text


def post(path: str, *, json_body: dict | None = None, data: dict | None = None, files: list | None = None) -> Any:
    form = {k: (str(v).lower() if isinstance(v, bool) else str(v)) for k, v in (data or {}).items() if v is not None}
    if json_body is not None:
        return request("POST", path, json=json_body).json()
    return request("POST", path, data=form, files=files or None).json()


def stream_events(batch_id: str, after: int = 0) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (event_name, data) from GET /batches/{id}/events until the server ends the stream.

    💡 SSE parsing in ten lines: `event:` names the event, `data:` lines carry the payload, and a blank
       line means "this event is complete". Lines starting with ':' are keep-alive comments.
    """
    url = f"{API_URL}/batches/{batch_id}/events"
    try:
        with httpx.stream("GET", url, params={"after": after}, timeout=httpx.Timeout(10.0, read=None)) as response:
            if not response.is_success:
                response.read()  # ⚠️ a streamed body must be read before .json() works
                _raise_for(response)
            name, data = "message", []
            for line in response.iter_lines():
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    name = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())
                elif not line and data:
                    yield name, json.loads("\n".join(data))
                    name, data = "message", []
    except httpx.TransportError as exc:
        raise _unreachable(exc) from exc
