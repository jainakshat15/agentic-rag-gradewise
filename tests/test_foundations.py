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


def _diagnostics_module():
    """Import setup/diagnostics.py by path (it isn't a package); importing it runs no checks."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("course_diagnostics", REPO_ROOT / "setup" / "diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("sk-or-v1-" + "0123456789abcdef" * 4, "sk-or-v1-… (73 chars)"),
        ('"sk-or-v1-' + "f" * 64 + '"', "sk-or-v1-… (73 chars)"),  # quotes are stripped before measuring
        ("sk-proj-" + "a" * 40, "sk-proj-… (48 chars)"),
        ("sk-ant-api03-" + "b" * 40, "sk-ant-api03-… (53 chars)"),
        ("lsv2_pt_" + "c" * 40, "lsv2_pt_… (48 chars)"),
        ("not-a-known-key-format-xyz", "… (26 chars)"),
        ("sk-or-v1-a", "… (10 chars)"),  # too short to show even the prefix
    ],
)
def test_diagnostics_mask_shows_only_prefix_and_length(key, expected):
    """⚠️ The mask must never print any character of the secret part of a key (rule: blur-free screenshots)."""
    masked = _diagnostics_module().Diagnostics._mask(key)
    assert masked == expected
    secret_part = key.strip("\"' ")[len(expected.split("…")[0]):]
    assert secret_part[-4:] not in masked.split("…")[1]


def test_diagnostics_key_line_never_prints_key_characters():
    """The console line for a well-formed OpenRouter key shows the public prefix and the length only."""
    module = _diagnostics_module()
    diag = module.Diagnostics()
    key = "sk-or-v1-" + "9f8e7d6c5b4a3210" * 4
    diag._check_key_format("OPENROUTER_API_KEY", key, "cheap")
    printed = "\n".join(diag.lines)
    assert "OPENROUTER_API_KEY is set (sk-or-v1-… (73 chars))" in printed
    assert key[9:] not in printed and key[-4:] not in printed
