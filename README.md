# 🎓 Agentic RAG with LangChain & LangGraph
### 6 weeks to build **GradeWise**: an AI agent that reads handwritten exams, grades them with evidence and writes report cards

![GradeWise: from handwritten answer sheet to report card](assets/hero.png)

_Looking at this in Cursor or VS Code? Right-click `README.md` in the Explorer on the left and choose **"Open Preview"** to see the formatted version._

---

## 👋 Welcome!

I couldn't be more excited to have you here. Over the next six weeks we build one serious, end-to-end agentic system together, and you'll understand every line of it.

Here's the story. **Asha** teaches *CS201 Systems Fundamentals* at the (fictional) **Northbridge Institute**. Every term she spends whole weekends marking handwritten exam scripts. By the end of this course, **GradeWise** will do the heavy lifting for her:

1. 📚 **Learn the course material**: ingest course PDFs into a RAG knowledge base
2. 📝 **Understand the exam**: turn the question paper, marking scheme and methodology into validated JSON
3. ✍️ **Read handwriting**: transcribe each student's answer sheet with a vision model, faithfully
4. ⚖️ **Grade with evidence**: an agent marks every answer against the rubric, citing the course material
5. 📊 **Write report cards**: store them, and let Asha *chat with the results*

And Asha always stays in control: GradeWise asks a human whenever it isn't sure.

---

## 🗺️ The 6-week roadmap

![Course roadmap](assets/roadmap/overview.png)

| Week | Theme | GradeWise milestone | Folder |
|:---:|---|---|---|
| **1** | Foundations & your first win | Exam paper → validated JSON | [week1](week1/README.md) |
| **2** | RAG: the knowledge base | Course PDFs → searchable, measured knowledge base | [week2](week2/README.md) |
| **3** | Agents & LangGraph | Self-correcting exam extraction with admin approval | [week3](week3/README.md) |
| **4** | Agentic RAG & reading handwriting | Evidence retriever + handwriting transcription | [week4](week4/README.md) |
| **5** | Grading agent & evaluation | End-to-end grading → report cards, checked against human marks | [week5](week5/README.md) |
| **6** | Ship it | FastAPI + Streamlit app, in Docker, deployed | [week6](week6/README.md) |

Every week runs on the same rhythm: **Days 1–4** teach the ideas (theory first, then hands-on), **Day 5 is project day** when GradeWise gains a new capability, and a **quiz** wraps up the week.

---

## 🛠️ The not-so-dreaded setup instructions

I've worked hard to make this as painless as possible. Pick your platform:

- 🪟 **Windows** people, your instructions are [here](setup/SETUP-PC.md)
- 🍎 **Mac** people, yours are [here](setup/SETUP-mac.md)
- 🐧 **Linux** people, yours are [here](setup/SETUP-linux.md)

When you're done, this one command checks everything and tells you exactly what (if anything) to fix:

```bash
uv run setup/diagnostics.py
```

Stuck? Open [setup/troubleshooting.ipynb](setup/troubleshooting.ipynb). It walks through the most common problems one cell at a time.

---

## 💰 API costs: please read me!

This course calls AI models through an API, which needs a key and a small spend. We set this up in the setup guides.

| | Expected spend |
|---|---|
| Whole course, default settings | **under $10** (most weeks cost cents) |
| Whole course with `BUDGET_MODE=true` | about $3 |
| Using free local models via Ollama | $0 (slower, lower quality) |

- OpenAI asks for a small minimum credit (about $5). **Set a monthly budget limit** in your account.
- Watch your spend at [platform.openai.com/usage](https://platform.openai.com/usage).
- Prefer Anthropic, Google Gemini or local models? Everything works with them. See [setup/SWAP_PROVIDER.md](setup/SWAP_PROVIDER.md).

---

## 🗣️ How this repo talks to you

This repo is designed to guide you at every step. Look out for these:

| You'll see | It means |
|---|---|
| A banner at the top of every `.py` file | Which week/day it belongs to, what it does, what came before and what uses it next |
| <img src="assets/callouts/important.png" width="24"> **Important** (orange) | A gotcha worth slowing down for |
| <img src="assets/callouts/exercise.png" width="24"> **Exercise** (amber) | Your turn: try it yourself |
| <img src="assets/callouts/business.png" width="24"> **Business value** (green) | How this is used in the real world |
| <img src="assets/callouts/stop.png" width="24"> **Stop & think** (red) | Predict the result before you run the cell |
| <img src="assets/callouts/resources.png" width="24"> **Resources** (blue) | Further reading |
| <img src="assets/callouts/gradewise.png" width="24"> **GradeWise** (violet) | Where this idea lives in our project |
| Green ✅ panels in your terminal | Something worked, plus what to do next |
| Red ❌ panels | Something broke, with the likely cause, the fix and where to read more |

In code comments: `# 💡` explains a concept, `# ⚠️` flags a pitfall, `# 🔁 RECAP` links back to an earlier lesson, `# 🧪 TRY` suggests an experiment, and `# 🚀 PRO` is a production tip beyond the course.

---

## 📁 What's where

```
├── setup/            🛠️ installation guides, diagnostics, troubleshooting
├── guides/           📘 self-study notebooks: command line, git, Python, async, APIs…
├── week1 … week6/    🎬 one folder per week: day1–5 notebooks, exercises, solutions, project code
├── gradewise/        🧱 the finished capstone package (what weeks 1–6 build up to), incl. the API and dashboard
├── data/             📚 knowledge base, exam pack, synthetic answer sheets, golden marks
├── quizzes/          📝 the weekly quizzes + the final quiz (also on Udemy)
├── slides/           🖼️ every week's slide deck (PowerPoint sources)
├── tests/            🧪 the free, offline test suite: uv run python -m pytest -q
├── scripts/          🔧 rebuild the course data, images and diagrams
├── assets/           🎨 banners, roadmaps, callout icons, diagrams and screenshots
├── docker/ · docker-compose.yml   🐳 run the whole app with one command (week 6)
├── output/           📦 everything GradeWise writes while you work (git-ignored)
└── community_contributions/  🤝 share your own extensions here
```

---

## 📚 Super useful resources

- 🖼️ The **slides** for every week: [slides/](slides/README.md)
- 📘 Self-study **guides**, from the command line to async Python: [guides/](guides/01_intro.ipynb)
- 🔗 Every link mentioned in the course, organised by week: [RESOURCES.md](RESOURCES.md)
- 🩺 Something broken? [setup/troubleshooting.ipynb](setup/troubleshooting.ipynb)
- 🤖 Using Cursor or Claude Code? The repo includes an [AGENTS.md](AGENTS.md) so your AI assistant understands the course too.

## 🔄 Keeping up to date

I update this repo as LangChain and the models evolve. Pull the latest at the start of each week:

```bash
git pull
uv sync
```

What changed and when is in [CHANGELOG.md](CHANGELOG.md).

## 🤝 Get in touch & contribute

- Questions? Post in the Udemy Q&A for the lecture you're on: I read them all.
- Built something cool on top of GradeWise? Share it in [community_contributions/](community_contributions/README.md) (see [CONTRIBUTING.md](CONTRIBUTING.md)).
- Let's connect on LinkedIn: _link coming soon_

---

### ⭐ Above all else

Have fun with it! There has never been a better time to learn agentic AI, and you're about to build something genuinely useful. Let's go. 🚀
