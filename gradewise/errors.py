"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 1 · Day 1: Errors that tell you how to fix them             │
│                                                                      │
│  What this file does                                                 │
│    Every GradeWise error carries four things: what happened, the     │
│    likely cause, the fix, and where to read more. In a notebook or   │
│    terminal, call `err.show()` to print it as a red panel.           │
│                                                                      │
│  ⏭ Used by: config checks, models.py, every pipeline from week 2 on │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from gradewise.console import failure


class GradeWiseError(Exception):
    """Base class. Subclasses fill in sensible defaults for cause/fix/help."""

    title = "GradeWise error"
    cause = ""
    fix = ""
    help_ref = "TROUBLESHOOTING: setup/troubleshooting.ipynb"

    def __init__(self, what: str, *, cause: str = "", fix: str = "", help_ref: str = "") -> None:
        self.what = what
        self.cause = cause or self.cause
        self.fix = fix or self.fix
        self.help_ref = help_ref or self.help_ref
        super().__init__(self.__str__())

    def __str__(self) -> str:
        lines = [self.what]
        if self.cause:
            lines.append(f"Likely cause: {self.cause}")
        if self.fix:
            lines.append(f"Fix: {self.fix}")
        if self.help_ref:
            lines.append(f"Help: {self.help_ref}")
        return "\n".join(lines)

    def show(self) -> None:
        """Print this error as a friendly red panel."""
        failure(self.title, self.what, self.cause, self.fix, self.help_ref)


class MissingAPIKeyError(GradeWiseError):
    title = "Missing API key"
    cause = "The .env file is missing, not in the repo root, not saved, or the key is misspelled."
    fix = "Copy .env.example to .env, paste your key, and SAVE the file. Then restart the kernel."
    help_ref = "Part 5 of setup/SETUP-PC.md (or -mac / -linux)"


class InvalidModelError(GradeWiseError):
    title = "Model setting looks wrong"
    cause = 'Models must be written as "provider:model", e.g. "openai:gpt-5.4-mini".'
    fix = "Check CHAT_MODEL / SMART_MODEL / VISION_MODEL in your .env file."
    help_ref = "setup/SWAP_PROVIDER.md"


class DataNotFoundError(GradeWiseError):
    title = "Course data not found"
    cause = "A file the lab expects isn't in the data/ folder."
    fix = "Run `git pull` to get the latest course files, and check data/README.md."
    help_ref = "data/README.md"
