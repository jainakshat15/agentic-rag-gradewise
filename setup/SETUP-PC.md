## 🎓 Agentic RAG with LangChain & LangGraph: Build GradeWise

# 🪟 Setup instructions for Windows

Welcome, Windows people! This guide gets your machine ready. It goes with the video **Week 1 · Day 1: "Windows Setup: Git, Cursor/VS Code, Clone and `uv sync`"**. Watch it alongside if you like seeing every click.

Each Part ends with a ✅ **Checkpoint**. Don't move on until the checkpoint passes: a problem fixed now saves an hour later.

> 💡 **If you get stuck at any point**, run `uv run setup/diagnostics.py` (or `python setup/diagnostics.py` if uv isn't installed yet). It checks everything and tells you exactly what to fix. Then see [troubleshooting.ipynb](troubleshooting.ipynb).

---

## ⚠️ Before we begin: the 4 Windows gotchas. Please read this!

These four catch more students than everything else combined. Two minutes now will save you a headache later. 😅

1. **Permissions.** Install things as yourself, into folders you own (like `C:\Users\you\projects`). Avoid `C:\Program Files` and OneDrive-synced folders for the repo. If an installer complains about permissions, right-click it and choose **Run as administrator**.

2. **Antivirus, firewall, VPN and corporate security.** These can block downloads or network calls to the AI providers. If installs hang or API calls time out, try temporarily pausing your VPN. On a **work laptop**, security policies ("Application Control" or "Smart App Control") can block newly downloaded programs and Python packages. You'll see *"An Application Control policy has blocked this file"*. Note which file is named and see [troubleshooting.ipynb](troubleshooting.ipynb) §4.

3. **The evil 260-character path limit.** Windows still limits file paths to 260 characters by default, and some Python packages have very deep folders. To lift the limit, open **PowerShell as Administrator** and run:
   ```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```
   Then **restart your computer**. Also, clone the repo into a *short* path such as `C:\Users\you\projects\`.

4. **Microsoft C++ Build Tools.** You probably won't need these: every package in this course installs ready-made on Windows. Install them only if an install error mentions `cl.exe` or `Microsoft Visual C++ 14.0 is required`. Then download the **Build Tools for Visual Studio** from https://visualstudio.microsoft.com/visual-cpp-build-tools/, tick **"Desktop development with C++"** in the installer, and install.

---

## Part 1: Install your code editor (Cursor or VS Code)

I use **VS Code** in the videos. **Cursor** (VS Code with AI built in) works identically for this course.

1. Download **Cursor** from https://cursor.com/ (or VS Code from https://code.visualstudio.com/)
2. Run the installer with the default options, then open it.
3. Install two extensions: click the **Extensions** icon in the left sidebar (four squares), search for and install:
   - **Python** (by Microsoft)
   - **Jupyter** (by Microsoft)

> 💡 Your editor may show a splash screen that looks different from mine: that's fine, they update it often.

✅ **Checkpoint:** Cursor/VS Code opens, and both extensions show as installed.

---

## Part 2: Install Git and clone the repo

1. In Cursor, open a terminal: **View → Terminal** (or <kbd>Ctrl</kbd>+<kbd>`</kbd>). It will be **PowerShell**.
2. Check whether Git is installed:
   ```powershell
   git --version
   ```
   If you see `git version 2.x`, skip to step 4.
3. If not: download Git from https://git-scm.com/downloads/win and run the installer, pressing **Next** on every screen (the defaults are perfect). Then **close and reopen Cursor** so it notices Git.
4. Go to your projects folder (create it first if needed):
   ```powershell
   mkdir C:\Users\$env:USERNAME\projects -Force
   cd C:\Users\$env:USERNAME\projects
   ```
5. Clone the course repo:
   ```powershell
   git clone https://github.com/<your-github-username>/agentic-rag-gradewise.git
   ```
   _(The exact URL is shown in the lecture resources and at the top of the GitHub page.)_
6. Open the project: **File → Open Folder…** → choose `C:\Users\you\projects\agentic-rag-gradewise`. When asked *"Do you trust the authors?"*, choose **Yes**.

✅ **Checkpoint:** the Explorer on the left shows folders like `setup`, `week1`, `gradewise` and the file `README.md`.

---

## Part 3: The amazing `uv`

`uv` is a super-fast tool that installs Python *and* every package the course needs, into a private environment just for this project. No Anaconda, no `pip install` marathons.

1. In the Cursor terminal (PowerShell), install uv:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. **Close and reopen the terminal** (the trash-can icon, then <kbd>Ctrl</kbd>+<kbd>`</kbd> again), then check:
   ```powershell
   uv --version
   ```
   If uv was already installed, make sure it's current: `uv self update`
3. Make sure you're in the repo folder (your prompt should end in `agentic-rag-gradewise>`), then run:
   ```powershell
   uv sync
   ```
   This downloads Python 3.12 if needed and installs everything. The first time takes a few minutes. ☕

✅ **Checkpoint:**
- a `.venv` folder now appears in the Explorer, and
- `uv run python --version` prints `Python 3.12.x`

### 💡 How to work with uv (read this once!)
| Instead of… | With uv you do… |
|---|---|
| `python my_script.py` | `uv run my_script.py` |
| `pip install some-package` | `uv add some-package` |
| activating a virtual environment | nothing: `uv run` handles it |

- You don't need to run `uv sync` often: `uv run` syncs automatically. Run it after every `git pull`, just to be safe.
- Please don't edit `pyproject.toml` or `uv.lock` by hand.
- uv's docs are excellent: https://docs.astral.sh/uv/

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

> 💡 **Other providers?** OpenAI, Anthropic (Claude), Google Gemini with their own keys, and free local models via Ollama all work with `MODEL_MODE=custom`. Finish this guide, then follow [SWAP_PROVIDER.md](SWAP_PROVIDER.md).

✅ **Checkpoint:** you have a key starting with `sk-or-` copied somewhere safe (a password manager, not a sticky note!).

---

## Part 5: The `.env` file

Your key lives in a file called `.env` in the repo root. It's already in `.gitignore`, so it can never be committed to GitHub by accident.

1. In the Cursor terminal, from the repo folder:
   ```powershell
   copy .env.example .env
   ```
2. Open `.env` in Cursor (it's in the Explorer, near the bottom).
3. Paste your key after `OPENROUTER_API_KEY=`, with **no quotes and no spaces**:
   ```
   OPENROUTER_API_KEY=sk-or-v1-abc123...
   ```
   That's all `cheap` mode needs. For the $0 route, also change these two lines near the top:
   ```
   MODEL_MODE=free
   DATA_SIZE=small
   ```
   (`DATA_SIZE=small` uses 2 PDFs, questions Q1–Q3 and 2 students: also handy in `cheap` mode for a quick run.)
4. **SAVE THE FILE** (<kbd>Ctrl</kbd>+<kbd>S</kbd>). Honestly, unsaved `.env` files are the #1 setup problem in every course I've seen. 🙂

> ⚠️ **Windows trap:** if you created the file another way (e.g. Notepad), Windows may have secretly named it `.env.txt`. In Explorer, enable **View → Show → File name extensions** to check. The name must be exactly `.env`.

✅ **Checkpoint:** run the diagnostics:
```powershell
uv run setup/diagnostics.py
```
Every section should show ✅ (a few ⚠️ warnings are OK: read them). Check 4 shows your `MODEL_MODE` and `DATA_SIZE`; check 9 confirms OpenRouter accepts your key (a free call that uses no tokens). The full report is saved to `setup/report.txt`.

---

## Part 6: Open your first notebook

1. In the Explorer, open `week1/day1.ipynb`
2. Top-right of the notebook, click **Select Kernel → Python Environments…** → choose the one marked **`.venv`** (path ends in `agentic-rag-gradewise\.venv\Scripts\python.exe`)
   - Don't see it? Press <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> → **Developer: Reload Window**, then try again.
3. Click on the first code cell and press <kbd>Shift</kbd>+<kbd>Enter</kbd>.

✅ **Checkpoint:** the setup cell prints a green ✅ panel. **You're ready for the course!** 🎉

---

## 🆘 Still stuck?

1. Run `uv run setup/diagnostics.py` and read every ❌ and its → hint
2. Work through [troubleshooting.ipynb](troubleshooting.ipynb)
3. Post in the Udemy Q&A for the setup lecture: attach `setup/report.txt` (it contains no secrets) and tell me what you tried

I really want you to succeed with this, so please reach out!
