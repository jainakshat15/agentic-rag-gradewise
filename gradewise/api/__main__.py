"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🚢 Week 6 · Day 1: `python -m gradewise.api` starts the server      │
│                                                                      │
│  Run : uv run python -m gradewise.api                                │
│        uv run python -m gradewise.api --demo --port 8000 --reload    │
│  Env : API_HOST (default 127.0.0.1) · API_PORT (8000)                │
│        GRADEWISE_DEMO=true → offline fakes, no API key needed        │
│  ⏭ Used by: docker/Dockerfile, the `gradewise serve api` command    │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from gradewise.console import banner, tip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m gradewise.api", description="Run the GradeWise API.")
    parser.add_argument("--host", default=os.environ.get("API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("API_PORT", "8000")))
    parser.add_argument("--demo", action="store_true", help="Offline demo mode: fake models, no API key")
    parser.add_argument("--reload", action="store_true", help="Restart when code changes (development)")
    args = parser.parse_args(argv)

    if args.demo:
        os.environ["GRADEWISE_DEMO"] = "true"  # 💡 read by gradewise.api.workspace when the app starts
    banner("GradeWise API", f"http://{args.host}:{args.port}  ·  docs at /docs")
    if args.host == "127.0.0.1":
        tip("Only this computer can reach the API. Use --host 0.0.0.0 inside Docker or on a server.")
    # 💡 An import string ("module:attribute"), not the app object, so --reload can re-import it.
    # ⚠️ Keep ONE worker process: the batch job registry lives in memory (see jobs.py).
    uvicorn.run("gradewise.api.main:app", host=args.host, port=args.port, reload=args.reload, workers=1)


if __name__ == "__main__":
    main()
