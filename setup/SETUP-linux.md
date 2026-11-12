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

Other providers (`MODEL_MODE=custom`): [SWAP_PROVIDER.md](SWAP_PROVIDER.md).

## Part 5: `.env`
```bash
cp .env.example .env
nano .env            # OPENROUTER_API_KEY=sk-or-v1-...  (no quotes, no spaces), then save
                     # $0 route: also MODEL_MODE=free and DATA_SIZE=small
uv run setup/diagnostics.py
```
✅ All sections ✅

## Part 6: First notebook
Open `week1/day1.ipynb` in Cursor/VS Code, select the `.venv` kernel, and run the first cell. Prefer the browser? `uv run jupyter lab`.

✅ A green panel. You're ready! 🎉
