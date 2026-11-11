"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🖼️ scripts/render_diagrams.py: Mermaid text → pictures              │
│                                                                      │
│  Notebook viewers in Cursor / VS Code show Mermaid as raw text, so   │
│  every diagram in this repo is shipped as a PNG instead.             │
│                                                                      │
│  1) markdown: finds ```mermaid blocks in notebooks and READMEs,      │
│     renders each to assets/diagrams/<id>.png and swaps the block     │
│     for the image (the source stays in a collapsible section).       │
│        uv run scripts/render_diagrams.py markdown                    │
│  2) graphs: renders the .mmd files captured from notebook runs       │
│     (see GRADEWISE_DIAGRAM_EXPORT in gradewise/display.py) into      │
│     assets/diagrams/graphs/<id>.png, which show_graph() picks up.    │
│        uv run scripts/render_diagrams.py graphs <folder-of-mmd>      │
│                                                                      │
│  Rendering is LOCAL with mermaid-cli + your installed Chrome, so no  │
│  diagram leaves your machine. Set MERMAID_CLI to mermaid-cli's       │
│  src/cli.js (v10.9 works with Node 16+) and MERMAID_PUPPETEER to a   │
│  JSON file like {"executablePath": "<path to chrome>"}.              │
│                                                                      │
│  💡 Learners never need to run this: the PNGs are committed. Course  │
│     maintainers re-run it after rebuilding any notebook.             │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gradewise.console import banner, step, success, warn  # noqa: E402
from gradewise.display import THEME_CONFIG, diagram_id  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIAGRAMS = ROOT / "assets" / "diagrams"
BLOCK = re.compile(r"```mermaid\s*\n(.*?)```", re.S)
SKIP_PARTS = {".venv", "output", ".ipynb_checkpoints", "node_modules"}


def render(code: str, target: Path) -> bool:
    """Render one diagram to PNG with mermaid-cli. Returns True on success."""
    if target.exists():
        return True
    cli = os.environ.get("MERMAID_CLI")
    puppeteer = os.environ.get("MERMAID_PUPPETEER")
    if not cli:
        warn("Set MERMAID_CLI to mermaid-cli's src/cli.js (see this file's header).")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src, cfg = Path(tmp) / "d.mmd", Path(tmp) / "config.json"
        src.write_text(code.strip() + "\n", encoding="utf-8")
        cfg.write_text(json.dumps(THEME_CONFIG), encoding="utf-8")
        cmd = ["node", cli, "-i", str(src), "-o", str(target), "-c", str(cfg), "-b", "#0B1020", "-s", "2", "-q"]
        if puppeteer:
            cmd += ["-p", puppeteer]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0 or not target.exists():
        warn(f"Couldn't render {target.name}: {result.stderr.strip()[-300:]}")
        return False
    return True


def _replacement(code: str, rel_png: str, *, notebook: bool) -> str:
    # In notebooks we keep the source as ```text so it never shows as raw Mermaid again;
    # in Markdown files we keep ```mermaid so GitHub still renders it inside <details>.
    fence = "text" if notebook else "mermaid"
    return (
        f"![Diagram]({rel_png})\n\n"
        f"<details><summary>🔎 Diagram source (Mermaid)</summary>\n\n```{fence}\n{code.strip()}\n```\n\n</details>\n"
    )


def convert_text(text: str, base: Path, *, notebook: bool) -> tuple[str, int]:
    count = 0

    def swap(match: re.Match) -> str:
        nonlocal count
        code = match.group(1)
        png = DIAGRAMS / f"{diagram_id(code)}.png"
        if not render(code, png):
            return match.group(0)
        count += 1
        return _replacement(code, os.path.relpath(png, base).replace("\\", "/"), notebook=notebook)

    return BLOCK.sub(swap, text), count


def markdown_mode() -> None:
    banner("Rendering Mermaid diagrams", "notebooks + READMEs → assets/diagrams/")
    totals: dict[str, int] = {}
    for path in sorted(ROOT.rglob("*")):
        if path.suffix not in {".ipynb", ".md"} or SKIP_PARTS & set(path.parts):
            continue
        if path.suffix == ".ipynb":
            nb = nbformat.read(path, as_version=4)
            changed = 0
            for cell in nb.cells:
                if cell.cell_type == "markdown" and "```mermaid" in cell.source:
                    cell.source, n = convert_text(cell.source, path.parent, notebook=True)
                    changed += n
            if changed:
                nbformat.write(nb, path)
        else:
            text = path.read_text(encoding="utf-8")
            if "```mermaid" not in text or "<summary>🔎 Diagram source" in text:
                continue
            new, changed = convert_text(text, path.parent, notebook=False)
            if changed:
                path.write_text(new, encoding="utf-8")
        if changed:
            step(f"{path.relative_to(ROOT)}: {changed} diagram(s)")
            totals[str(path.relative_to(ROOT))] = changed
    success("Diagrams rendered", {"Files changed": len(totals), "Diagrams": sum(totals.values())})


def graphs_mode(folder: Path) -> None:
    banner("Rendering captured graph diagrams", f"{folder} → assets/diagrams/graphs/")
    done = 0
    for mmd in sorted(folder.glob("*.mmd")):
        code = mmd.read_text(encoding="utf-8")
        if render(code, DIAGRAMS / "graphs" / f"{diagram_id(code)}.png"):
            done += 1
    success("Graph pictures rendered", {"Graphs": done})


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "markdown":
        markdown_mode()
    elif len(sys.argv) >= 3 and sys.argv[1] == "graphs":
        graphs_mode(Path(sys.argv[2]))
    else:
        print(__doc__)
