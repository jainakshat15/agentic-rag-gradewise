## 🎓 Agentic RAG with LangChain & LangGraph: Build GradeWise

# 🍎 Setup instructions for Mac

Welcome, Mac people! This takes about **15–25 minutes** and goes with the video **Week 1 · Day 1: "Mac Setup: Git, Cursor/VS Code, Clone and `uv sync`"**.

Each Part ends with a ✅ **Checkpoint**. Please don't skip them!

> 💡 **Stuck?** Run `uv run setup/diagnostics.py` (or `python3 setup/diagnostics.py` if uv isn't installed yet), then see [troubleshooting.ipynb](troubleshooting.ipynb).

---

## ⚠️ Before we begin: 3 Mac gotchas

1. **Command Line Tools.** Git and compilers come from Apple's Command Line Tools. If any command below pops up *"…requires the command line developer tools"*, click **Install** and wait for it to finish.
2. **Apple Silicon vs Intel.** Everything in this course works on both. If you use Homebrew, make sure it matches your chip (`uname -m` shows `arm64` on Apple Silicon).
3. **Anaconda.** If you have Anaconda installed and your prompt starts with `(base)`, run `conda deactivate` before working on this course (and consider `conda config --set auto_activate_base false`). It can clash with uv.

---

## Part 1: Install your code editor (Cursor or VS Code)

1. Download **Cursor** from https://cursor.com/ (or VS Code from https://code.visualstudio.com/), drag it into **Applications**, and open it.
2. Install the **Python** and **Jupyter** extensions (Microsoft): click the **Extensions** icon (four squares) in the sidebar, then search.

✅ **Checkpoint:** both extensions show as installed.

---

## Part 2: Git and the repo

1. Open a terminal in Cursor: **View → Terminal** (<kbd>Ctrl</kbd>+<kbd>`</kbd>).
2. Check Git: `git --version`. If macOS offers to install the developer tools, accept, wait, then run it again.
3. Create a projects folder and clone:
   ```bash
   mkdir -p ~/projects && cd ~/projects
   git clone https://github.com/<your-github-username>/agentic-rag-gradewise.git
   ```
4. **File → Open Folder…** → `~/projects/agentic-rag-gradewise` → trust the authors.

✅ **Checkpoint:** the Explorer shows `setup`, `week1`, `gradewise` and `README.md`.

---

## Part 3: The amazing `uv`

1. Install uv:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Close and reopen the terminal**, then check: `uv --version` (already had it? `uv self update`)
3. From the repo folder:
   ```bash
   uv sync
   ```

✅ **Checkpoint:** a `.venv` folder appears, and `uv run python --version` prints `Python 3.12.x`.

| Instead of… | With uv you do… |
|---|---|
| `python my_script.py` | `uv run my_script.py` |
| `pip install some-package` | `uv add some-package` |
| activating a virtual environment | nothing: `uv run` handles it |

---

## Part 4: Your OpenAI API key

1. Sign in at https://platform.openai.com/ → **Settings → Billing** → add the minimum credit (about $5). The whole course costs under $10.
2. **Settings → Limits** → set a monthly budget.
3. **API keys → Create new secret key** → copy it right away.

> 💡 Prefer Anthropic, Gemini (free tier) or local Ollama models? See [SWAP_PROVIDER.md](SWAP_PROVIDER.md) after finishing this guide.

---

## Part 5: The `.env` file

```bash
cp .env.example .env
```
Open `.env` in Cursor (files starting with `.` are hidden in Finder but visible in Cursor), then paste:
```
OPENAI_API_KEY=sk-proj-abc123...
```
No quotes, no spaces. **Save with <kbd>Cmd</kbd>+<kbd>S</kbd>!**

✅ **Checkpoint:**
```bash
uv run setup/diagnostics.py
```
All sections ✅ (warnings ⚠️ are worth reading but usually OK).

---

## Part 6: Open your first notebook

1. Open `week1/day1.ipynb`
2. **Select Kernel → Python Environments…** → choose **`.venv`** (`…/agentic-rag-gradewise/.venv/bin/python`). Not listed? <kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> → **Developer: Reload Window**.
3. Run the first cell with <kbd>Shift</kbd>+<kbd>Enter</kbd>.

✅ **Checkpoint:** a green ✅ panel appears. **You're ready!** 🎉

---

## 🆘 Still stuck?
Run the diagnostics, work through [troubleshooting.ipynb](troubleshooting.ipynb), then post in the Udemy Q&A with `setup/report.txt` attached (it contains no secrets).
