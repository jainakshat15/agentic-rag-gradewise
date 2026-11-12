"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧪 tests/test_foundations.py: checks the gradewise foundations      │
│                                                                      │
│  Run:  uv run pytest                                                 │
│  💡 These tests never call a paid API, so they're free to run.       │
╰──────────────────────────────────────────────────────────────────────╯
"""

import pytest

from gradewise.config import REPO_ROOT
from gradewise.errors import InvalidModelError, MissingAPIKeyError
from gradewise.models import _check


def test_repo_root_is_found():
    assert (REPO_ROOT / "pyproject.toml").exists()


def test_model_without_provider_is_rejected():
    with pytest.raises(InvalidModelError):
        _check("gpt-6-luna")


def test_missing_key_gives_friendly_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("gradewise.models.settings.anthropic_api_key", None)
    with pytest.raises(MissingAPIKeyError) as exc:
        _check("anthropic:claude-haiku-5-5")
    assert "Fix:" in str(exc.value)


def test_ollama_needs_no_key():
    assert _check("ollama:llama3.2") == "ollama:llama3.2"


def test_diagnostics_is_stdlib_only():
    """setup/diagnostics.py must work even when package installation failed."""
    import ast

    tree = ast.parse((REPO_ROOT / "setup" / "diagnostics.py").read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])
    }
    third_party = imported - set(__import__("sys").stdlib_module_names) - {"__future__", ""}
    assert not third_party, f"diagnostics.py imports non-stdlib modules: {third_party}"
