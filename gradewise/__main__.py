"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧱 gradewise/__main__.py: `python -m gradewise` entry point         │
│                                                                      │
│      uv run python -m gradewise --help   (same as `uv run gradewise`)│
│                                                                      │
│  💡 Why two ways? On some locked-down (often corporate) Windows      │
│     machines, security policies block the small `gradewise.exe`      │
│     launcher that uv creates. Running the package as a module avoids │
│     that launcher entirely.                                          │
│                                                                      │
│  ⏭ Built on: gradewise/cli.py                                       │
╰──────────────────────────────────────────────────────────────────────╯
"""

from gradewise.cli import app

app()
