"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🚢 Week 6 · Day 1: Where the API keeps things (and demo mode)       │
│                                                                      │
│  What this file does                                                 │
│    One small object, Workspace, answers "where does this go?" for    │
│    every route: the SQLite file, the vector store, report cards,     │
│    checkpoints and uploads. There are two of them:                   │
│                                                                      │
│      real  → output/           (shared with the CLI and notebooks)   │
│      demo  → output/demo/      (fake marks never mix with real ones) │
│                                                                      │
│    Uploads always land in output/uploads/, NEVER in data/: data/ is  │
│    the course's own material, tracked by git.                        │
│                                                                      │
│  ⏭ Used by: main.py, jobs.py                                        │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from gradewise.config import OUTPUT_DIR, settings
from gradewise.errors import GradeWiseError

_TRUE = {"1", "true", "yes", "on"}


def demo_from_env() -> bool:
    """GRADEWISE_DEMO=true (in the environment or .env) makes demo mode the API's default.

    💡 Every request can still override it with its own `demo` flag (the dashboard's sidebar toggle).
    """
    raw = os.environ.get("GRADEWISE_DEMO")
    if raw is None:
        raw = _dotenv_value("GRADEWISE_DEMO")
    return (raw or "").strip().lower() in _TRUE


def has_provider_key() -> bool:
    """True if every key your MODEL_MODE needs is configured (cheap and free → OPENROUTER_API_KEY).

    💡 A mode that only uses local Ollama models needs no key at all.
    """
    import os

    for env in settings.required_keys:
        secret = getattr(settings, env.lower(), None)
        if not ((secret and secret.get_secret_value()) or os.getenv(env)):
            return False
    return True


def _dotenv_value(name: str) -> str | None:
    """Read one key from the repo's .env (settings ignores keys it doesn't know, like GRADEWISE_DEMO)."""
    from gradewise.config import REPO_ROOT

    env = REPO_ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == name:
            return value.split("#", 1)[0].strip().strip('"').strip("'")
    return None


class InvalidNameError(GradeWiseError):
    title = "That name can't be used"
    cause = "Names become folder names, so they may only contain letters, digits, '-' and '_'."
    fix = 'Pick something like "cs_midterm" or "cs201".'
    help_ref = "week6/project/README.md"


_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def safe_name(value: str, what: str = "name") -> str:
    """⚠️ Never build a path from raw user input: "../../etc" would escape the folder. Allow-list instead."""
    if not _SAFE.match(value or ""):
        raise InvalidNameError(f"{value!r} is not a valid {what}.")
    return value


def safe_filename(filename: str | None, default: str = "upload.pdf") -> str:
    """Keep only the last path part of an uploaded file's name, with risky characters replaced."""
    name = Path(filename or default).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")
    return name or default


@dataclass(frozen=True)
class Workspace:
    """Every path the API writes to, for one mode (real or demo)."""

    root: Path
    uploads: Path
    demo: bool

    @property
    def db(self) -> Path:
        return self.root / "gradewise.db"

    @property
    def vector_dir(self) -> Path:
        return self.root / "vector_db"

    @property
    def batch_checkpoints(self) -> Path:
        return self.root / "api_batch_checkpoints.db"

    @property
    def exam_checkpoints(self) -> Path:
        return self.root / "api_exam_checkpoints.db"

    def reports_dir(self, exam_id: str) -> Path:
        return self.root / "reports" / safe_name(exam_id, "exam ID")

    def upload_dir(self, *parts: str) -> Path:
        folder = self.uploads.joinpath("demo" if self.demo else "real", *parts)
        folder.mkdir(parents=True, exist_ok=True)
        return folder


def make_workspaces(output_dir: Path | None = None) -> dict[bool, Workspace]:
    """{False: real workspace, True: demo workspace} under `output_dir` (default output/)."""
    base = Path(output_dir) if output_dir else OUTPUT_DIR
    uploads = base / "uploads"
    return {
        False: Workspace(root=base, uploads=uploads, demo=False),
        True: Workspace(root=base / "demo", uploads=uploads, demo=True),
    }
