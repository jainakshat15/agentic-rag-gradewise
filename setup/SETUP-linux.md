## 🎓 Agentic RAG with LangChain & LangGraph: Build GradeWise

# 🐧 Setup instructions for Linux

Linux people, you've probably done most of this before, so here's the fast version. Each Part still ends with a ✅ **Checkpoint**.

> 💡 **Stuck?** `uv run setup/diagnostics.py` (or `python3 setup/diagnostics.py`), then [troubleshooting.ipynb](troubleshooting.ipynb).

## ⚠️ Gotchas
- **Build essentials:** a few packages may compile code. Debian/Ubuntu: `sudo apt install build-essential git curl`. Fedora: `sudo dnf groupinstall "Development Tools" && sudo dnf install git curl`.
- **Don't use the system Python or `sudo pip`.** uv installs its own Python 3.12 just for this project.
- **WSL users:** clone inside the Linux filesystem (`~/projects`), not under `/mnt/c/…`. It's much faster and avoids permission oddities.

## Part 1: Editor
Install **Cursor** (https://cursor.com/, AppImage) or **VS Code** (https://code.visualstudio.com/), plus the **Python** and **Jupyter** extensions.

## Part 2: Clone
```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/<your-github-username>/agentic-rag-gradewise.git
cd agentic-rag-gradewise
```
✅ `ls` shows `setup  week1  gradewise  README.md …`

## Part 3: uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL          # reload your shell so `uv` is on PATH
uv sync
```
✅ `uv run python --version` → `Python 3.12.x`

## Part 4: OpenAI key
https://platform.openai.com/ → add credit (about $5; the course costs under $10) → set a budget limit → create a key. Other providers: [SWAP_PROVIDER.md](SWAP_PROVIDER.md).

## Part 5: `.env`
```bash
cp .env.example .env
nano .env            # OPENAI_API_KEY=sk-proj-...  (no quotes, no spaces), then save
uv run setup/diagnostics.py
```
✅ All sections ✅

## Part 6: First notebook
Open `week1/day1.ipynb` in Cursor/VS Code, select the `.venv` kernel, and run the first cell. Prefer the browser? `uv run jupyter lab`.

✅ A green panel. You're ready! 🎉
