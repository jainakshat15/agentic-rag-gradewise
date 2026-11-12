"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 1 · Day 1: The repo's "voice"                               │
│                                                                      │
│  What this file does                                                 │
│    Gives every script and notebook the same friendly, colourful      │
│    output: success panels, step headers, tips and cost readouts.     │
│    When GradeWise talks to you, it talks through these functions.    │
│                                                                      │
│  Use it                                                              │
│    from gradewise.console import success, step, tip                  │
│    success("Ingestion complete", {"Chunks": 412}, next_step="...")   │
│                                                                      │
│  ⏭ Used by: everything                                              │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

# 🎨 The same palette as the slide deck (see COURSE_PLAN §8), so the terminal,
#    the notebooks and the videos all feel like one product.
THEME = Theme(
    {
        "primary": "bold #7C5CFF",
        "accent": "#22D3EE",
        "ok": "bold #34D399",
        "warn": "bold #FBBF24",
        "err": "bold #F87171",
        "muted": "#9AA3C7",
    }
)

# ⚠️ Windows gotcha: older terminals use the "cp1252" code page, which can't print
#    emoji like ✅, and Python crashes with UnicodeEncodeError. We switch the output
#    to UTF-8 (and replace anything unprintable) so GradeWise never crashes on a tick.
for _stream in (sys.stdout, sys.stderr):
    if _stream and (getattr(_stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # 💡 e.g. Jupyter's streams: they're already UTF-8

console = Console(theme=THEME)


def _kv_table(rows: Mapping[str, Any]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="accent", no_wrap=True)
    table.add_column()
    for key, value in rows.items():
        table.add_row(str(key), str(value))
    return table


def banner(title: str, subtitle: str = "") -> None:
    """A big header, used at the top of scripts: `banner('Week 2 · Day 5', 'Ingestion')`."""
    body = f"[primary]{title}[/]" + (f"\n[muted]{subtitle}[/]" if subtitle else "")
    console.print(Panel.fit(body, border_style="primary", padding=(0, 2)))


def model_banner_text() -> str:
    """The two banner lines: MODEL_MODE (+ models) and DATA_SIZE, e.g.

    🧠 MODEL_MODE=cheap: gpt-6-luna (chat · grading · vision) · embeddings text-embedding-3-small · ...
    📦 DATA_SIZE=full: 4 PDFs · all 10 questions (15 parts) · 10 students
    """
    from gradewise import datasets
    from gradewise.config import get_settings

    return f"{get_settings().describe_models()}\n{datasets.describe()}"


def model_banner() -> None:
    """Print which models and how much data this run uses (notebooks call it in their setup cell)."""
    from gradewise.config import get_settings

    s = get_settings()
    keys = ", ".join(s.required_keys) or "no key (local models)"
    hint = f"🔑 needs {keys} · switch with MODEL_MODE / DATA_SIZE in .env"
    console.print(
        Panel.fit(
            f"[primary]{model_banner_text()}[/]\n[muted]{hint}[/]",
            border_style="primary",
            padding=(0, 2),
        )
    )
    from gradewise import datasets

    if s.is_free and not datasets.is_small():
        warn("MODEL_MODE=free allows about 50 requests a day: set DATA_SIZE=small for a full run of the course.")


def step(message: str) -> None:
    """One line per step of a pipeline: `step('Loading PDFs…')`."""
    console.print(f"[accent]▸[/] {message}")


def success(title: str, details: Mapping[str, Any] | None = None, next_step: str = "") -> None:
    """The green panel you see when something finishes."""
    parts: list[Any] = []
    if details:
        parts.append(_kv_table(details))
    if next_step:
        parts.append(f"\n[muted]⏭  Next:[/] {next_step}")
    from rich.console import Group

    console.print(
        Panel(
            Group(*parts) if parts else "",
            title=f"[ok]✅ GradeWise · {title}[/]",
            title_align="left",
            border_style="ok",
        )
    )


def warn(message: str) -> None:
    """Something to watch, but not fatal."""
    console.print(f"[warn]⚠️  {message}[/]")


def tip(message: str) -> None:
    """A friendly aside: `tip('Set DATA_SIZE=small to run on less data')`."""
    console.print(f"[accent]💡 {message}[/]")


def failure(title: str, what: str, cause: str = "", fix: str = "", help_ref: str = "") -> None:
    """The red panel. Used by gradewise.errors so every error explains how to fix itself."""
    rows: dict[str, str] = {"What happened": what}
    if cause:
        rows["Likely cause"] = cause
    if fix:
        rows["Fix"] = fix
    if help_ref:
        rows["Help"] = help_ref
    console.print(Panel(_kv_table(rows), title=f"[err]❌ {title}[/]", title_align="left", border_style="err"))
