## 🎓 Agentic RAG with LangChain & LangGraph: Build GradeWise

# 🍎 Setup instructions for Mac

Welcome, Mac people! This guide goes with the video **Week 1 · Day 1: "Mac Setup: Git, Cursor/VS Code, Clone and `uv sync`"**.

Each Part ends with a ✅ **Checkpoint**. Please don't skip them!

> 💡 **Stuck?** Run `uv run setup/diagnostics.py` (or `python3 setup/diagnostics.py` if uv isn't installed yet), then see [troubleshooting.ipynb](troubleshooting.ipynb).

---

## ⚠️ Before we begin: 3 Mac gotchas

1. **Command Line Tools.** Git and compilers come from Apple's Command Line Tools. If any command below pops up *"…requires the command line developer tools"*, click **Install** and wait for it to finish.
2. **Apple Silicon vs Intel.** Everything in this course works on both, and you don't need Homebrew for it. To check your chip, run `uname -m`: it prints `arm64` on Apple Silicon and `x86_64` on Intel.
3. **Anaconda.** If you have Anaconda installed and your prompt starts with `(base)`, run `conda deactivate` before working on this course. To stop `(base)` from turning on in every new terminal, run `conda config --set auto_activate false` (older conda versions call this setting `auto_activate_base`). It can clash with uv, and `setup/diagnostics.py` warns you if a conda environment is still active.

---

## Part 1: Install your code editor (Cursor or VS Code)

1. Download **Cursor** from https://cursor.com/ (or VS Code from https://code.visualstudio.com/), drag it into **Applications**, and open it.
2. Install the **Python** and **Jupyter** extensions (Microsoft): click the **Extensions** icon (four squares) in the sidebar, then search.

✅ **Checkpoint:** both extensions show as installed.

---

## Part 2: Git and the repo

1. Open a terminal in Cursor: **View → Terminal** (<kbd>Ctrl</kbd>+<kbd>`</kbd>).
2. Check Git: `git --version`. If macOS offers to install the developer tools, accept, wait, then run it again. (You can also start that install yourself with `xcode-select --install`.)
3. Create a projects folder and clone:
   ```bash
   mkdir -p ~/projects && cd ~/projects
   git clone https://github.com/jainakshat15/agentic-rag-gradewise.git
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

## Part 4: Your OpenRouter API key

GradeWise reaches every model through **[OpenRouter](https://openrouter.ai)**, so you need **one key**, `OPENROUTER_API_KEY`, for both modes. One line in `.env`, `MODEL_MODE`, then picks the models:
- `cheap` (the default): OpenAI's **gpt-6-luna**, $0.10 / $0.50 per 1M tokens. Needs a few dollars of OpenRouter credits.
- `free`: **openrouter/free**, OpenRouter's free-models router. $0 and no credits, but rate-limited.

1. Sign up at https://openrouter.ai (a Google or GitHub login works).
2. Go to https://openrouter.ai/keys → **Create Key**. Name it `gradewise-course`, then **copy it right away**: it starts with `sk-or-` and you can't see it again later.
3. **For `cheap` mode, add credits** at https://openrouter.ai/settings/credits. A few dollars is plenty: week 1 costs a few cents, and GradeWise prints the cost of each run. OpenRouter adds a small fee when you buy credits. **`free` mode needs no credits.**
4. **Set a credit limit on the key** (on the keys page) so there are never surprises. Agentic systems are less predictable than ordinary code, which is the point, but it means it's wise to cap spending.

> ⚠️ **Free-mode caveats** (`MODEL_MODE=free`):
> 1. **Rate limits:** 20 requests a minute and **50 requests a day** until your account has ever bought $10 of credits (then 1,000 a day).
> 2. **Use it with `DATA_SIZE=small`:** a full class run makes hundreds of calls.
> 3. **Quality and data:** the router picks a different free model each call, so results vary and an occasional structured-output failure is normal (GradeWise retries). Free providers may log prompts or train on them: fine for this course's fictional data, never OK for real student data. In your [OpenRouter privacy settings](https://openrouter.ai/settings/privacy) you may need to enable **"free endpoints that may train on inputs"**.
>
> Details, and the optional stronger grader: [SWAP_PROVIDER.md](SWAP_PROVIDER.md).

> 💡 Prefer OpenAI, Anthropic or Gemini with their own keys, or local Ollama models (`MODEL_MODE=custom`)? See [SWAP_PROVIDER.md](SWAP_PROVIDER.md) after finishing this guide.

---

## Part 5: The `.env` file

```bash
cp .env.example .env
```
Open `.env` in Cursor (files starting with `.` are hidden in Finder but visible in Cursor), then paste your key after `OPENROUTER_API_KEY=`:
```
OPENROUTER_API_KEY=sk-or-v1-abc123...
```
That's all `cheap` mode needs. For the $0 route, also change these two lines near the top:
```
MODEL_MODE=free
DATA_SIZE=small
```
(`DATA_SIZE=small` uses 2 PDFs, questions Q1–Q3 and 2 students: also handy in `cheap` mode for a quick run.)

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

✅ **Checkpoint:** the first cell prints `✅ Great: you're using the course's .venv kernel…`; run Part 2 and you get a green **Your settings** panel. **You're ready!** 🎉

---

## 🆘 Still stuck?
Run the diagnostics, work through [troubleshooting.ipynb](troubleshooting.ipynb), then post in the Udemy Q&A with `setup/report.txt` attached (it contains no secrets).
