"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🩺 setup/diagnostics.py: "is my machine ready for the course?"      │
│                                                                      │
│  Run it from the repo root:                                          │
│      uv run setup/diagnostics.py                                     │
│  or, if uv itself is the problem:                                    │
│      python setup/diagnostics.py                                     │
│                                                                      │
│  It runs 10 checks, prints ✅ / ⚠️ / ❌ for each, and writes the     │
│  full details to setup/report.txt. If you ask for help, please       │
│  attach that file: it contains no secrets (keys are masked).         │
│                                                                      │
│  💡 This file deliberately uses ONLY the Python standard library,    │
│     so it still works when package installation has failed.         │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import datetime as _dt
import importlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

# ⚠️ Windows terminals sometimes can't print emoji: switch output to UTF-8 first.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_FILE = Path(__file__).resolve().parent / "report.txt"

# Packages the course can't live without (import name → friendly name)
KEY_PACKAGES = {
    "langchain": "langchain",
    "langgraph": "langgraph",
    "langchain_openai": "langchain-openai",
    "langchain_openrouter": "langchain-openrouter",
    "langchain_chroma": "langchain-chroma",
    "chromadb": "chromadb",
    "pymupdf": "pymupdf",
    "pydantic": "pydantic",
    "dotenv": "python-dotenv",
    "rich": "rich",
    "ipykernel": "ipykernel",
}

# 💰 MODEL_MODE → (key it needs, models). Keep in step with gradewise/config.py (this file can't import it).
MODES = {
    "cheap": (
        "OPENROUTER_API_KEY",
        "openrouter:openai/gpt-6-luna (chat · grading · vision) + openrouter:openai/text-embedding-3-small",
    ),
    "free": (
        "OPENROUTER_API_KEY",
        "openrouter:openrouter/free (the router picks a free model per call) + "
        "openrouter:nvidia/nemotron-3.5-embed-1b:free",
    ),
}
PROVIDER_KEYS = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}
CUSTOM_DEFAULTS = {
    "CHAT_MODEL": "openrouter:openai/gpt-6-luna",
    "SMART_MODEL": "openrouter:openai/gpt-6-luna",
    "VISION_MODEL": "openrouter:openai/gpt-6-luna",
    "EMBEDDING_MODEL": "openrouter:openai/text-embedding-3-small",
}
QUOTES = "\"' "  # 💡 what we strip from values: quotes and spaces

# Hosts we need to reach (host → why)
HOSTS = {
    "openrouter.ai": "OpenRouter API (MODEL_MODE=cheap and free)",
    "api.smith.langchain.com": "LangSmith tracing (optional)",
    "pypi.org": "Installing packages with uv",
    "github.com": "git pull for course updates",
}


class Diagnostics:
    """Runs every check, collecting results for the console and report.txt."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    # ── output helpers ───────────────────────────────────────────────────
    def log(self, message: str = "") -> None:
        self.lines.append(message)

    def ok(self, message: str) -> None:
        print(f"   ✅ {message}")
        self.log(f"OK      {message}")

    def warning(self, message: str, fix: str = "") -> None:
        print(f"   ⚠️  {message}" + (f"\n       → {fix}" if fix else ""))
        self.log(f"WARNING {message}" + (f" | FIX: {fix}" if fix else ""))
        self.warnings.append(message)

    def error(self, message: str, fix: str = "") -> None:
        print(f"   ❌ {message}" + (f"\n       → {fix}" if fix else ""))
        self.log(f"ERROR   {message}" + (f" | FIX: {fix}" if fix else ""))
        self.errors.append(message)

    def section(self, number: int, title: str) -> None:
        print(f"\n[{number}/10] {title}")
        self.log(f"\n=== {number}. {title} ===")

    # ── the checks ───────────────────────────────────────────────────────
    def check_system(self) -> None:
        self.section(1, "System")
        self.ok(f"{platform.system()} {platform.release()} ({platform.machine()})")
        self.log(f"Platform: {platform.platform()}")
        major, minor = sys.version_info[:2]
        if (major, minor) >= (3, 12):
            self.ok(f"Python {platform.python_version()} at {sys.executable}")
        else:
            self.error(
                f"Python {platform.python_version()} is too old (need 3.12+)",
                "Run this script with `uv run setup/diagnostics.py` so uv picks Python 3.12",
            )
        free_gb = shutil.disk_usage(REPO_ROOT).free / 1e9
        if free_gb < 5:
            self.warning(f"Only {free_gb:.1f} GB free disk space", "Free up at least 5 GB")
        else:
            self.ok(f"{free_gb:.0f} GB free disk space")
        if platform.system() == "Windows":
            self._check_windows_long_paths()

    def _check_windows_long_paths(self) -> None:
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem")
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
            if value == 1:
                self.ok("Windows long paths are enabled")
            else:
                self.warning(
                    "Windows long paths are DISABLED (the 260-character limit)",
                    "See 'The 4 Windows gotchas', item 3, in setup/SETUP-PC.md",
                )
        except OSError:
            self.warning("Couldn't read the Windows long-paths setting", "See setup/SETUP-PC.md")
        if len(str(REPO_ROOT)) > 60:
            self.warning(
                f"The repo lives in a long path ({len(str(REPO_ROOT))} characters)",
                "A short path like C:\\Users\\you\\projects\\ avoids Windows path-length errors",
            )

    def check_repo_files(self) -> None:
        self.section(2, "Course files")
        expected = ["pyproject.toml", "uv.lock", ".python-version", "gradewise", "week1", "setup"]
        missing = [name for name in expected if not (REPO_ROOT / name).exists()]
        if missing:
            self.error(
                f"Missing from the repo root: {', '.join(missing)}",
                "Make sure you are running from the cloned repo folder, then `git pull`",
            )
        else:
            self.ok(f"All course files present in {REPO_ROOT}")

    def check_git(self) -> None:
        self.section(3, "Git")
        git = shutil.which("git")
        if not git:
            self.error("git is not installed", "Install from https://git-scm.com/downloads")
            return
        version = self._run(["git", "--version"])
        self.ok(version or "git found")
        if not (REPO_ROOT / ".git").exists():
            self.warning(
                "This folder isn't a git clone, so you won't be able to `git pull` updates",
                "Clone the repo with `git clone` instead of downloading a ZIP",
            )
            return
        remote = self._run(["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"])
        if remote:
            self.ok(f"Remote origin: {remote}")
        else:
            self.warning("No 'origin' remote configured", "Fine if you cloned it locally")
        status = self._run(["git", "-C", str(REPO_ROOT), "status", "-sb"])
        if status and "behind" in status.splitlines()[0]:
            self.warning("Your copy is behind the course repo", "Run `git pull` then `uv sync`")

    def check_env_file(self) -> None:
        self.section(4, ".env file and API keys")
        env_path = REPO_ROOT / ".env"
        if not env_path.exists():
            if (REPO_ROOT / ".env.txt").exists():
                self.error(
                    "Found .env.txt instead of .env (Windows hid the extension)",
                    "Rename it to exactly `.env` (see Part 5 of setup/SETUP-PC.md)",
                )
            else:
                self.error(
                    "No .env file in the repo root",
                    "Copy .env.example to .env and paste your key (Part 5 of the setup guide)",
                )
            return
        self.ok(".env file found")
        values = self._parse_env(env_path)
        mode, needed = self._mode(values)
        if mode not in ("cheap", "free", "custom"):
            self.error(
                f"MODEL_MODE={mode!r} isn't a mode GradeWise knows",
                "Use MODEL_MODE=cheap (gpt-6-luna), free (openrouter/free) or custom",
            )
            return
        size = (values.get("DATA_SIZE") or "full").strip(QUOTES).lower() or "full"
        if size not in ("full", "small"):
            self.error(f"DATA_SIZE={size!r} isn't valid", "Use DATA_SIZE=full or DATA_SIZE=small")
        if "BUDGET_MODE" in values:
            self.log("BUDGET_MODE is no longer used (MODEL_MODE replaced it): you can delete that line")
        self.ok(f"MODEL_MODE={mode} · DATA_SIZE={size}")
        if mode in MODES:
            self.log(f"  Models: {MODES[mode][1]}")
        else:
            for var, default in CUSTOM_DEFAULTS.items():
                self.log(f"  {var}={values.get(var) or default}")
        if size == "small":
            self.log("  Small data: 2 PDFs · Q1–Q3 · 2 students (DATA_SIZE=full for everything)")
        self.log(f"  This mode needs: {', '.join(needed) or 'no key (local models only)'}")
        for name in needed:
            self._check_key_format(name, values.get(name, ""), mode)
        others = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "LANGSMITH_API_KEY")
        for name in others:
            if name not in needed and values.get(name):
                self.ok(f"{name} is set ({self._mask(values[name])})")

    def _check_key_format(self, name: str, key: str, mode: str) -> None:
        """Is the key this mode needs present, and does it look like a key?"""
        if not key:
            fix = {
                "OPENROUTER_API_KEY": "Create a key at https://openrouter.ai/keys, paste it into .env, SAVE",
            }.get(name, "Paste your key into .env and SAVE the file")
            self.error(f"{name} is empty, but MODEL_MODE={mode} needs it", fix)
            return
        problems = []
        if key != key.strip():
            problems.append("it has spaces around it")
        if key.startswith(('"', "'")):
            problems.append("it is wrapped in quotes")
        prefix = {
            "OPENROUTER_API_KEY": "sk-or-",
            "OPENAI_API_KEY": "sk-",
            "GOOGLE_API_KEY": "AI",
            "ANTHROPIC_API_KEY": "sk-ant-",
        }.get(name)
        if prefix and not key.strip(QUOTES).startswith(prefix):
            problems.append(f"it doesn't start with '{prefix}'")
        if problems:
            example = {"OPENROUTER_API_KEY": "sk-or-v1-abc123...", "OPENAI_API_KEY": "sk-proj-abc123..."}.get(
                name, "..."
            )
            self.warning(f"{name} looks unusual: {', '.join(problems)}", f"The line should look like: {name}={example}")
        else:
            self.ok(f"{name} is set ({self._mask(key)})")

    @staticmethod
    def _mode(values: dict[str, str]) -> tuple[str, list[str]]:
        """(MODEL_MODE, [key variables it needs]) from the .env values."""
        mode = (values.get("MODEL_MODE") or os.getenv("MODEL_MODE") or "cheap").strip(QUOTES).lower() or "cheap"
        if mode in MODES:
            return mode, [MODES[mode][0]]
        needed: list[str] = []
        for var, default in CUSTOM_DEFAULTS.items():
            provider = (values.get(var) or default).split(":", 1)[0]
            key = PROVIDER_KEYS.get(provider)
            if key and key not in needed:
                needed.append(key)
        return mode, needed

    def check_conflicting_envs(self) -> None:
        self.section(5, "Conflicting Python environments")
        conda = os.getenv("CONDA_DEFAULT_ENV")
        if conda:
            self.warning(
                f"An Anaconda environment is active ('{conda}')",
                "Run `conda deactivate` (repeat until none is active), then use `uv run`",
            )
        else:
            self.ok("No Anaconda environment active")
        venv = os.getenv("VIRTUAL_ENV")
        if venv and Path(venv).resolve() != (REPO_ROOT / ".venv").resolve():
            self.warning(
                f"A different virtual environment is active: {venv}",
                "Run `deactivate`, then use `uv run` from the repo root",
            )

    def check_virtualenv(self) -> None:
        self.section(6, "The course environment (.venv) and key packages")
        in_repo_venv = Path(sys.prefix).resolve() == (REPO_ROOT / ".venv").resolve()
        if in_repo_venv:
            self.ok("Running inside the course's .venv")
        elif (REPO_ROOT / ".venv").exists():
            self.warning(
                "A .venv exists, but this script isn't running inside it",
                "Use `uv run setup/diagnostics.py` instead of `python ...`",
            )
            return
        else:
            self.error("No .venv folder yet", "Run `uv sync` in the repo root (Part 3 of setup)")
            return
        missing = []
        for module, name in KEY_PACKAGES.items():
            try:
                mod = importlib.import_module(module)
                version = getattr(mod, "__version__", "")
                self.log(f"  {name} {version}")
            except Exception as exc:  # noqa: BLE001 - we want every failure
                missing.append(name)
                self.log(f"  {name} FAILED: {exc!r}")
        if missing:
            self.error(f"Packages failed to import: {', '.join(missing)}", "Run `uv sync` again")
        else:
            self.ok(f"All {len(KEY_PACKAGES)} key packages import correctly")

    def check_uv(self) -> None:
        self.section(7, "uv")
        uv = shutil.which("uv")
        if uv:
            self.ok(self._run(["uv", "--version"]) or "uv found")
        else:
            self.error("uv is not on your PATH", "Install it (Part 3 of setup), then reopen the terminal")

    def check_network(self) -> None:
        self.section(8, "Network")
        for host, why in HOSTS.items():
            try:
                socket.create_connection((host, 443), timeout=5).close()
                self.ok(f"Can reach {host}")
            except OSError as exc:
                self.warning(
                    f"Can't reach {host} ({why})",
                    "Check VPN/firewall/proxy settings (Windows gotcha 2 in setup/SETUP-PC.md)",
                )
                self.log(f"  {host}: {exc!r}")
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        if proxy:
            self.log(f"HTTPS proxy in use: {proxy}")

    def check_key_works(self) -> None:
        self.section(9, "Does your key actually work? (free call, no tokens used)")
        values = self._parse_env(REPO_ROOT / ".env") if (REPO_ROOT / ".env").exists() else {}
        mode, needed = self._mode(values)
        if "OPENROUTER_API_KEY" not in needed:
            self.log(f"Skipped: MODEL_MODE={mode} doesn't use OpenRouter (custom keys aren't tested here)")
            return
        key = (values.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")).strip(QUOTES)
        if not key:
            self.warning("Skipped: no OpenRouter key to test", "Paste it into .env first (section 4 above)")
            return
        # 💡 GET /key describes the key itself: free to call, no tokens, and the key goes in a header
        #    (never in the URL, never printed).
        request = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                info = json.loads(response.read().decode("utf-8")).get("data", {})
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                self.error(
                    f"OpenRouter rejected the key ({exc.code})", "Create a new key at https://openrouter.ai/keys"
                )
            elif exc.code == 429:
                self.warning("OpenRouter says 429 (too many requests)", "Wait a minute and run this again")
            else:
                self.warning(f"OpenRouter returned HTTP {exc.code}")
            return
        except (OSError, ValueError) as exc:
            self.warning("Couldn't contact OpenRouter to test the key")
            self.log(f"  {exc!r}")
            return
        self.ok("OpenRouter accepted your key")
        free_tier = info.get("is_free_tier")
        if free_tier is not None:
            self.log(f"  Account has bought credits: {'no' if free_tier else 'yes'}")
        if mode == "cheap" and free_tier:
            self.warning(
                "MODEL_MODE=cheap needs OpenRouter credits, and this account has none yet",
                "Add a few dollars at https://openrouter.ai/settings/credits, or use MODEL_MODE=free",
            )
        if mode == "free":
            self.log("  Free models: about 50 requests a day (1,000 once you've bought $10 of credits ever)")

    def check_openai_key_works(self) -> None:
        """💡 The old name, kept so older notes still work."""
        self.check_key_works()

    def check_extras(self) -> None:
        self.section(10, "Extras: write access, LangSmith, Ollama")
        try:
            with tempfile.NamedTemporaryFile(dir=REPO_ROOT, delete=True):
                pass
            self.ok("The repo folder is writable")
        except OSError:
            self.error("Can't write to the repo folder", "Move the repo somewhere you own")
        values = self._parse_env(REPO_ROOT / ".env") if (REPO_ROOT / ".env").exists() else {}
        if values.get("LANGSMITH_TRACING", "").lower() == "true":
            if values.get("LANGSMITH_API_KEY"):
                self.ok("LangSmith tracing is on")
            else:
                self.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty")
        else:
            self.log("LangSmith tracing off (optional until week 5)")
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2).close()
            self.ok("Ollama is running locally (free model route available)")
        except OSError:
            self.log("Ollama not running (optional)")

    # ── utilities ────────────────────────────────────────────────────────
    @staticmethod
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    @staticmethod
    def _parse_env(path: Path) -> dict[str, str]:
        values: dict[str, str] = {}
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                name, _, value = line.partition("=")
                values[name.strip()] = value.split(" #")[0].strip()
        return values

    @staticmethod
    def _mask(secret: str) -> str:
        secret = secret.strip("\"' ")
        return f"{secret[:7]}…{secret[-4:]}" if len(secret) > 12 else "***"

    # ── run everything ───────────────────────────────────────────────────
    def run(self) -> int:
        print("\n🩺 GradeWise course diagnostics")
        print("   Checking your machine is ready for 'Agentic RAG with LangChain & LangGraph'")
        self.log(f"GradeWise diagnostics, {_dt.datetime.now():%Y-%m-%d %H:%M}")
        for check in (
            self.check_system,
            self.check_repo_files,
            self.check_git,
            self.check_env_file,
            self.check_conflicting_envs,
            self.check_virtualenv,
            self.check_uv,
            self.check_network,
            self.check_key_works,
            self.check_extras,
        ):
            try:
                check()
            except Exception as exc:  # noqa: BLE001 - a broken check must not stop the rest
                self.error(f"The check '{check.__name__}' itself crashed: {exc!r}")

        print("\n" + "─" * 70)
        if self.errors:
            print(f"❌ {len(self.errors)} problem(s) to fix, {len(self.warnings)} warning(s).")
            print("   Fix the ❌ items above (each has a → hint), then run this again.")
            print("   Still stuck? Open setup/troubleshooting.ipynb")
        elif self.warnings:
            print(f"⚠️  Ready, with {len(self.warnings)} warning(s) worth a look.")
            print("   ⏭  Next: open week1/day1.ipynb")
        else:
            print("✅ All checks passed. You're ready for the course!")
            print("   ⏭  Next: open week1/day1.ipynb")
        REPORT_FILE.write_text("\n".join(self.lines), encoding="utf-8")
        print(f"📄 Full report written to {REPORT_FILE.relative_to(REPO_ROOT)}\n")
        return 1 if self.errors else 0


if __name__ == "__main__":
    sys.exit(Diagnostics().run())
