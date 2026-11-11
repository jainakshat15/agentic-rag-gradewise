"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🖼️ gradewise/display.py: show diagrams as pictures, not text         │
│                                                                      │
│  What this file does                                                 │
│    LangGraph describes any graph as Mermaid text, but the notebook   │
│    viewers in Cursor / VS Code don't draw Mermaid. These helpers     │
│    show it as a picture instead:                                     │
│                                                                      │
│      show_graph(graph)    → draw a LangGraph graph / create_agent    │
│      show_mermaid(text)   → draw any Mermaid diagram                 │
│                                                                      │
│  How a picture is found, in order:                                   │
│    1. a pre-rendered PNG shipped with the course (assets/diagrams/   │
│       graphs/): works offline, in the course colours                 │
│    2. LangGraph's own draw_mermaid_png(), which asks the free        │
│       mermaid.ink service to draw it (needs the internet)            │
│    3. the Mermaid text itself, with a tip to paste it into           │
│       https://mermaid.live                                           │
│                                                                      │
│  💡 Changed a graph in a lab? Its picture won't be in the course     │
│     folder, so step 2 or 3 kicks in. That's expected.                │
│  ⏮ Built on: week3/day3.ipynb   ⏭ Used by: weeks 3–6 notebooks      │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from gradewise.config import REPO_ROOT

GRAPH_IMAGES = REPO_ROOT / "assets" / "diagrams" / "graphs"

# 🎨 The course palette (same as the slides and the Streamlit app), used by
#    scripts/render_diagrams.py when it pre-renders the shipped pictures.
THEME_CONFIG: dict[str, Any] = {
    "theme": "base",
    "themeVariables": {
        "background": "#0B1020",
        "primaryColor": "#141B34",
        "primaryTextColor": "#E6E9F5",
        "primaryBorderColor": "#7C5CFF",
        "secondaryColor": "#1B2550",
        "secondaryTextColor": "#E6E9F5",
        "secondaryBorderColor": "#22D3EE",
        "tertiaryColor": "#0F1630",
        "tertiaryTextColor": "#E6E9F5",
        "tertiaryBorderColor": "#2A335C",
        "lineColor": "#22D3EE",
        "textColor": "#E6E9F5",
        "mainBkg": "#141B34",
        "nodeBorder": "#7C5CFF",
        "clusterBkg": "#0F1630",
        "clusterBorder": "#2A335C",
        "titleColor": "#E6E9F5",
        "edgeLabelBackground": "#0B1020",
        "fontFamily": "Segoe UI, Inter, Helvetica, Arial, sans-serif",
        "fontSize": "16px",
    },
}

# 🚀 PRO (course maintainers): set this to a folder and every diagram shown in a
#    notebook run is saved there as .mmd, ready for scripts/render_diagrams.py.
EXPORT_ENV = "GRADEWISE_DIAGRAM_EXPORT"


def diagram_id(code: str) -> str:
    """A short, stable fingerprint of a Mermaid diagram (used as its file name)."""
    return hashlib.sha1(code.strip().encode("utf-8")).hexdigest()[:12]


def _export(code: str) -> None:
    folder = os.getenv(EXPORT_ENV)
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / f"{diagram_id(code)}.mmd").write_text(code.strip() + "\n", encoding="utf-8")


def show_mermaid(code: str, *, width: int | None = None) -> None:
    """Display a Mermaid diagram as an image in a notebook."""
    from IPython.display import Image, display

    _export(code)
    shipped = GRAPH_IMAGES / f"{diagram_id(code)}.png"
    if shipped.exists():
        display(Image(filename=str(shipped), width=width))
        return
    try:
        if os.getenv(EXPORT_ENV) or os.getenv("GRADEWISE_OFFLINE_DIAGRAMS"):
            # ⚠️ Capture runs and offline mode never send diagrams to a web service.
            raise RuntimeError("online drawing disabled")
        from langchain_core.runnables.graph_mermaid import draw_mermaid_png

        display(Image(data=draw_mermaid_png(code, background_color="#0B1020"), width=width))
    except Exception as exc:  # noqa: BLE001 - offline must never crash a lesson
        from gradewise.console import tip

        print(code)
        tip(f"Couldn't draw the picture ({type(exc).__name__}). Paste the text above into https://mermaid.live")


def show_graph(graph: Any, *, xray: bool | int = False, width: int | None = None) -> None:
    """Draw a compiled LangGraph graph, or a create_agent agent, as an image.

    🔁 RECAP: create_agent returns a compiled LangGraph graph, so this works for agents too.
    """
    show_mermaid(graph.get_graph(xray=xray).draw_mermaid(), width=width)
