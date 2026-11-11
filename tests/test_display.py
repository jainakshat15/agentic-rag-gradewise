"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧪 tests/test_display.py: diagrams show as pictures, never as text  │
│                                                                      │
│  Run:  uv run python -m pytest tests/test_display.py -q              │
│  💡 Free and offline: no diagram is ever sent to a web service.       │
╰──────────────────────────────────────────────────────────────────────╯
"""

import json
import re
from pathlib import Path

import nbformat
import pytest

from gradewise.config import REPO_ROOT
from gradewise.display import GRAPH_IMAGES, diagram_id, show_mermaid


def _notebooks():
    return [p for p in REPO_ROOT.rglob("*.ipynb") if not {".venv", "output"} & set(p.parts)]


def test_no_raw_mermaid_left_in_notebook_markdown():
    """Cursor / VS Code show ```mermaid blocks as raw text, so notebooks must use images."""
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in _notebooks()
        for cell in nbformat.read(p, as_version=4).cells
        if cell.cell_type == "markdown" and "```mermaid" in cell.source
    ]
    assert not offenders, f"Run `uv run scripts/render_diagrams.py markdown`; raw Mermaid in: {offenders}"


def test_no_code_cell_prints_mermaid_text():
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in _notebooks()
        for cell in nbformat.read(p, as_version=4).cells
        if cell.cell_type == "code" and re.search(r"print\([^)]*draw_mermaid\(\)", cell.source)
    ]
    assert not offenders, f"Use gradewise.display.show_graph(...) instead: {offenders}"


def test_every_diagram_image_link_exists():
    missing = []
    for p in _notebooks():
        for cell in nbformat.read(p, as_version=4).cells:
            for rel in re.findall(r"!\[Diagram\]\(([^)]+)\)", cell.source):
                if not (p.parent / rel).exists():
                    missing.append(f"{p.relative_to(REPO_ROOT)} → {rel}")
    assert not missing, missing


def test_shipped_graph_picture_is_used_offline(monkeypatch, capsys):
    shown = []
    monkeypatch.setattr("IPython.display.display", lambda obj: shown.append(obj))
    code = next(GRAPH_IMAGES.glob("*.png"), None)
    if code is None:
        pytest.skip("no shipped graph pictures yet")
    # find the Mermaid text whose id matches a shipped picture: we only need the lookup path
    monkeypatch.setattr("gradewise.display.diagram_id", lambda _code: code.stem)
    show_mermaid("graph TD; A-->B")
    assert shown and capsys.readouterr().out == ""


def test_offline_fallback_prints_text_and_never_crashes(monkeypatch, capsys):
    monkeypatch.setenv("GRADEWISE_OFFLINE_DIAGRAMS", "1")
    show_mermaid("graph TD; Unique_" + diagram_id(json.dumps(str(Path.cwd()))) + "-->B")
    out = capsys.readouterr().out
    assert "graph TD" in out and "mermaid.live" in out
