"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🎨 Week 6 · Day 2: The dashboard's look                             │
│                                                                      │
│  What this file does                                                 │
│    A little CSS on top of the theme in .streamlit/config.toml, plus  │
│    a handful of tiny "components" (plain HTML strings) every page    │
│    shares: hero(), stat_cards(), pill(), bar_chart(), table()...     │
│    One place for the look = every page feels like one product.       │
│                                                                      │
│  💡 Charts and tables are drawn as HTML/SVG on purpose: no pandas or │
│     pyarrow needed, so the UI runs even where Windows blocks their   │
│     DLLs, and it matches the report cards pixel for pixel.           │
│     🧪 TRY: swap bar_chart() for st.bar_chart on your machine.       │
│  ⏭ Used by: gradewise/ui/app.py and every page in pages/            │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Sequence
from html import escape

import streamlit as st

NAVY = "#0B1020"
SURFACE = "#141B34"
RAISED = "#1A2244"
BORDER = "#262F55"
VIOLET = "#7C5CFF"
CYAN = "#22D3EE"
GREEN = "#34D399"
AMBER = "#FBBF24"
RED = "#F87171"
TEXT = "#E6E9F5"
MUTED = "#9AA3C7"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
/* ⚠️ Don't restyle every element: Streamlit's icons are a font too (Material Symbols), and would turn into text. */
html, body, .stMarkdown, .stMarkdown p, h1, h2, h3, h4, h5, label, input, textarea {{
  font-family: 'Inter', system-ui, sans-serif; }}
code, pre, .gw-log {{ font-family: 'JetBrains Mono', ui-monospace, monospace !important; }}
.block-container {{ padding-top: 2.2rem; max-width: 1200px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}

.gw-hero {{
  position: relative; overflow: hidden; border-radius: 18px; padding: 26px 30px; margin-bottom: 18px;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(124,92,255,.35), transparent 60%),
              radial-gradient(800px 300px at 100% 100%, rgba(34,211,238,.18), transparent 60%), {SURFACE};
  border: 1px solid {BORDER};
}}
.gw-kicker {{ color: {CYAN}; font-weight: 700; font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; }}
.gw-hero h1 {{ margin: 6px 0 4px; font-size: 2.05rem; font-weight: 800; color: {TEXT}; padding: 0; }}
.gw-hero p {{ color: {MUTED}; margin: 0; font-size: 1.02rem; max-width: 820px; }}

.gw-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px;
  margin: 6px 0 18px; }}
.gw-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 16px; }}
.gw-card .label {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }}
.gw-card .value {{ color: {TEXT}; font-size: 1.7rem; font-weight: 800; margin-top: 2px; }}
.gw-card .hint {{ color: {MUTED}; font-size: .82rem; }}

.gw-pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
  border: 1px solid currentColor; margin: 0 4px 4px 0; white-space: nowrap; }}
.gw-pill.violet {{ color: {VIOLET}; background: rgba(124,92,255,.12); }}
.gw-pill.cyan {{ color: {CYAN}; background: rgba(34,211,238,.10); }}
.gw-pill.green {{ color: {GREEN}; background: rgba(52,211,153,.10); }}
.gw-pill.amber {{ color: {AMBER}; background: rgba(251,191,36,.10); }}
.gw-pill.red {{ color: {RED}; background: rgba(248,113,113,.10); }}
.gw-pill.muted {{ color: {MUTED}; background: rgba(154,163,199,.08); }}

.gw-flow {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px; margin: 8px 0 20px; }}
.gw-step {{ flex: 1 1 0; min-width: 104px; background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px;
  padding: 12px 14px; }}
.gw-step.done {{ border-color: rgba(52,211,153,.55); }}
.gw-step.wait {{ border-color: rgba(251,191,36,.65); box-shadow: 0 0 0 3px rgba(251,191,36,.10); }}
.gw-step .icon {{ font-size: 1.4rem; }}
.gw-step .name {{ font-weight: 700; color: {TEXT}; margin-top: 2px; }}
.gw-step .sub {{ color: {MUTED}; font-size: .8rem; }}
.gw-arrow {{ align-self: center; color: {VIOLET}; font-weight: 800; }}

.gw-answer {{ background: #0F1530; border-left: 3px solid {CYAN}; border-radius: 8px; padding: 10px 14px;
  color: {TEXT}; white-space: pre-wrap; font-size: .95rem; }}
.gw-crit {{ border-bottom: 1px solid rgba(38,47,85,.6); padding: 8px 0; }}
.gw-crit .top {{ display: flex; justify-content: space-between; gap: 8px; font-weight: 700; color: {TEXT}; }}
.gw-crit .why {{ color: {TEXT}; margin-top: 4px; font-size: .9rem; }}
.gw-muted {{ color: {MUTED}; font-size: .88rem; }}
.gw-log {{ background: #0A0F22; border: 1px solid {BORDER}; border-radius: 12px; padding: 10px 12px;
  font-size: .8rem; line-height: 1.55; max-height: 380px; overflow-y: auto; color: {TEXT}; }}
.gw-log .stage {{ color: {VIOLET}; }} .gw-log .who {{ color: {CYAN}; }} .gw-log .t {{ color: {MUTED}; }}

table.gw-table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
table.gw-table th {{ text-align: left; color: {MUTED}; font-weight: 600; font-size: .75rem; text-transform: uppercase;
  letter-spacing: .06em; border-bottom: 1px solid {BORDER}; padding: 8px 10px; }}
table.gw-table td {{ border-bottom: 1px solid rgba(38,47,85,.6); padding: 8px 10px; color: {TEXT};
  vertical-align: top; }}
table.gw-table tr:hover td {{ background: rgba(124,92,255,.06); }}

[data-testid="stSidebar"] .gw-brand {{ font-weight: 800; font-size: 1.25rem; color: {TEXT}; }}
[data-testid="stSidebar"] .gw-brand span {{ color: {VIOLET}; }}
div[data-testid="stProgress"] > div > div > div > div {{ background: linear-gradient(90deg, {VIOLET}, {CYAN}); }}
</style>
"""


def inject() -> None:
    """Add the CSS to the page. Call once per run (app.py does it before any page draws)."""
    st.html(CSS)


def html(markup: str) -> None:
    """Draw raw HTML. ⚠️ escape() anything that came from a user or a model before putting it in here."""
    st.html(markup)


def hero(title: str, subtitle: str = "", kicker: str = "GradeWise") -> None:
    html(
        f'<div class="gw-hero"><div class="gw-kicker">{escape(kicker)}</div>'
        f"<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>"
    )


def pill(text: str, colour: str = "violet") -> str:
    return f'<span class="gw-pill {colour}">{escape(text)}</span>'


def stat_cards(cards: Iterable[tuple[str, str] | tuple[str, str, str]]) -> None:
    """A responsive row of KPI cards: [(label, value) or (label, value, hint), ...]."""
    parts = []
    for card in cards:
        label, value, hint = (*card, "")[:3]
        parts.append(
            f'<div class="gw-card"><div class="label">{escape(label)}</div>'
            f'<div class="value">{escape(str(value))}</div><div class="hint">{escape(hint)}</div></div>'
        )
    html(f'<div class="gw-cards">{"".join(parts)}</div>')


def flow(steps: Sequence[tuple[str, str, str, str]]) -> None:
    """The pipeline diagram: [(icon, name, subtitle, state)], state = "done" | "wait" | ""."""
    parts = []
    for i, (icon, name, sub, state) in enumerate(steps):
        if i:
            parts.append('<div class="gw-arrow">→</div>')
        parts.append(
            f'<div class="gw-step {state}"><div class="icon">{icon}</div>'
            f'<div class="name">{escape(name)}</div><div class="sub">{escape(sub)}</div></div>'
        )
    html(f'<div class="gw-flow">{"".join(parts)}</div>')


def table(headers: Sequence[str], rows: Iterable[Sequence[str]], *, raw: bool = False) -> None:
    """A themed HTML table. Cells are escaped unless raw=True (then YOU must have escaped them)."""
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell if raw else escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows
    )
    html(f'<table class="gw-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    max_value: float | None = None,
    suffix: str = "",
    colours: Sequence[str] | None = None,
    height: int = 220,
) -> None:
    """A small vertical bar chart as inline SVG (crisp, themed, and no extra dependencies)."""
    if not values:
        return
    n = len(values)
    top = max_value or max(values) or 1
    width, pad, label_h = max(320, 64 * n), 28, 34
    bar_w = (width - pad * 2) / n * 0.62
    plot_h = height - label_h - 24
    bars = []
    for i, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = pad + (width - pad * 2) / n * (i + 0.19)
        h = max(2.0, plot_h * (value / top))
        y = 18 + plot_h - h
        colour = colours[i % len(colours)] if colours else "url(#gw-grad)"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="6" fill="{colour}"/>'
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" fill="{TEXT}" font-size="12" font-weight="700" '
            f'text-anchor="middle">{value:g}{escape(suffix)}</text>'
            f'<text x="{x + bar_w / 2:.1f}" y="{height - 12}" fill="{MUTED}" font-size="12" '
            f'text-anchor="middle">{escape(str(label))}</text>'
        )
    svg = (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" preserveAspectRatio="xMidYMid meet" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="Inter, Segoe UI, sans-serif">'
        f'<defs><linearGradient id="gw-grad" x1="0" y1="1" x2="0" y2="0">'
        f'<stop offset="0" stop-color="{VIOLET}"/><stop offset="1" stop-color="{CYAN}"/></linearGradient></defs>'
        f'<line x1="{pad}" y1="{18 + plot_h}" x2="{width - pad}" y2="{18 + plot_h}" stroke="{BORDER}"/>'
        f"{''.join(bars)}</svg>"
    )
    # ⚠️ st.html sanitises its input and drops inline <svg>, so the chart travels as an <img> data URL instead.
    src = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
    img = f'<img src="{src}" style="width:100%" alt="bar chart"/>'
    html(f'<div class="gw-card" style="padding:10px 12px">{img}</div>')


def band_colour(band: str) -> str:
    return {"A": "green", "B": "cyan", "C": "violet", "D": "amber"}.get(band, "red")
