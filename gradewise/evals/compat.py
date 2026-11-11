"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 5: Importing pandas on locked-down Windows machines         │
│                                                                      │
│  What: `pd = import_pandas()`. Exactly `import pandas as pd`, plus   │
│        one fallback for machines where Windows blocks pyarrow.       │
│  ⏭ Used by: gradewise/evals, gradewise/reports/qa_agent.py          │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import importlib.abc
import sys


class _BlockPyArrow(importlib.abc.MetaPathFinder):
    """Make `import pyarrow` fail cleanly, so pandas carries on as if pyarrow were not installed."""

    def find_spec(self, name, path, target=None):  # noqa: ARG002
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError(f"{name} is disabled: its DLLs are blocked on this machine")
        return None


def import_pandas():
    """Import pandas, falling back to "pandas without pyarrow" if pyarrow's compiled files are blocked.

    ⚠️ Windows gotcha (the same one gradewise/db.py works around for SQLAlchemy): with Windows
       "Application Control" / WDAC, freshly installed .pyd files can be blocked, and `import pandas`
       dies with "DLL load failed while importing _compute". pandas only NEEDS pyarrow for a few extras,
       so we hide pyarrow and import pandas again. Nothing GradeWise does uses pyarrow.
    """
    try:
        import pandas
    except ImportError as exc:
        if "DLL load failed" not in str(exc):
            raise
        for module in [m for m in sys.modules if m.split(".")[0] in ("pandas", "pyarrow")]:
            del sys.modules[module]  # 💡 forget the half-finished import before trying again
        sys.meta_path.insert(0, _BlockPyArrow())
        import pandas
    return pandas
