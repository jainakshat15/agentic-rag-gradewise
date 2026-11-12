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
    cause = (
        "MODEL_MODE=cheap and MODEL_MODE=free both need ONE key: OPENROUTER_API_KEY (MODEL_MODE=custom needs "
        "the key of each provider you picked). Or the .env file is missing, not in the repo root, not saved, "
        "or the key is misspelled."
    )
    fix = (
        "Create a key at https://openrouter.ai/keys, copy .env.example to .env, paste it after "
        "OPENROUTER_API_KEY=, and SAVE the file. Then restart the kernel."
    )
    help_ref = "Part 5 of setup/SETUP-PC.md (or -mac / -linux), and setup/SWAP_PROVIDER.md"


class FreeLimitReachedError(GradeWiseError):
    title = "Free daily limit reached"
    cause = (
        "MODEL_MODE=free uses OpenRouter's free models, which allow about 50 requests a day (1,000 a day once "
        "your account has ever bought $10 of credits). A full class run makes hundreds of calls."
    )
    fix = (
        "Use DATA_SIZE=small in .env, and try again tomorrow; or switch to MODEL_MODE=cheap "
        "(a few dollars of OpenRouter credits)."
    )
    help_ref = "setup/SWAP_PROVIDER.md (free-mode caveats)"


class NoCreditsError(GradeWiseError):
    title = "No OpenRouter credits"
    cause = "MODEL_MODE=cheap pays per call, and your OpenRouter account has no credits left (HTTP 402)."
    fix = (
        "Add a few dollars at https://openrouter.ai/settings/credits (OpenRouter adds a small fee when you buy), "
        "or set MODEL_MODE=free in .env (with DATA_SIZE=small)."
    )
    help_ref = "setup/SWAP_PROVIDER.md"


class InvalidModelError(GradeWiseError):
    title = "Model setting looks wrong"
    cause = 'Models must be written as "provider:model", e.g. "openrouter:openai/gpt-6-luna".'
    fix = (
        "Check MODEL_MODE in .env (cheap, free or custom). Only with MODEL_MODE=custom are "
        "CHAT_MODEL / SMART_MODEL / VISION_MODEL / EMBEDDING_MODEL used: check those too."
    )
    help_ref = "setup/SWAP_PROVIDER.md"


class DataNotFoundError(GradeWiseError):
    title = "Course data not found"
    cause = "A file the lab expects isn't in the data/ folder."
    fix = "Run `git pull` to get the latest course files, and check data/README.md."
    help_ref = "data/README.md"
