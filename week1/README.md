![Week 1](../assets/roadmap/week1.png)

# Week 1: Foundations & Your First Win

> **This week, GradeWise learns to understand an exam.** By Friday, it turns Emma's question paper, marking scheme and methodology into a clean, validated **JSON array**: the backbone everything else builds on.

## 🗓️ The week at a glance

| Day | Notebook | Videos | What you'll do |
|:---:|---|---|---|
| 1 | [day1.ipynb](day1.ipynb) | Welcome demo · 6-week roadmap · setup overview · Windows / Mac setup · API key & diagnostics | Get set up, make your first LLM call, and try a first taste of AI grading |
| 2 | [day2.ipynb](day2.ipynb) | How LLMs work: tokens, context windows, temperature & cost · the LangChain 1.x ecosystem · first calls with `init_chat_model` | Count tokens and dollars for the exam pack; invoke, stream, batch and swap models |
| 3 | [day3.ipynb](day3.ipynb) | Prompting for extraction · structured output with Pydantic · validation, retries & self-correction | Get reliable JSON for one exam question, and a loop that fixes its own mistakes |
| 4 | [day4.ipynb](day4.ipynb) | Multimodal: images & PDF pages · tools & tool calling | Send a PDF page as an image content block; write `@tool`s and a tool loop by hand |
| 5 | [day5.ipynb](day5.ipynb) | 🧱 GradeWise architecture · exam schema → JSON · week wrap-up | **Project day:** the exam pack → `ExamPaper` JSON, diffed field by field against the answer key |

🎬 **17 lectures** this week, plus the labs at your own pace.

## 🧱 This week's GradeWise milestone
```bash
uv run week1/project/extract_exam.py
```
Reads `data/exams/cs_midterm/` and writes `output/cs_midterm.json`: all 10 questions as **15 markable entries** (sub-parts like `Q7a` and `Q7b` are marked separately), **50 marks**, every criterion accounted for. Day 5 then compares it with the answer key, `data/exams/cs_midterm/expected_exam.json`. More in [project/README.md](project/README.md).

## ⚙️ One key and two switches in `.env`, before your first run
Get **one key**, `OPENROUTER_API_KEY`, at [openrouter.ai/keys](https://openrouter.ai/keys). It works for both `cheap` and `free`.

| Setting | Options | What it does |
|---|---|---|
| `MODEL_MODE` | `cheap` (default) · `free` · `custom` | `cheap`: `openai/gpt-6-luna` through OpenRouter; buy a few dollars of credits (OpenRouter adds a small fee). `free`: `openrouter/free`, $0 and no credits, but only 50 requests a day (1,000 once you've bought $10 of credits), so use it with `DATA_SIZE=small`; the model varies per call, and free providers may log prompts (fine for our fictional data). `custom`: your own `*_MODEL` lines, for any provider |
| `DATA_SIZE` | `full` (default) · `small` | `small`: 2 PDFs, questions Q1–Q3 and 2 students, for quick, cheap runs if you know the ideas already. (The exam extraction is still one call: the subset is applied after.) |

`uv run setup/diagnostics.py` prints your mode, its models, your data size and which key it needs.

## 💰 Expected API cost this week
**A few cents** in the default `cheap` mode, and **$0** with `MODEL_MODE=free`. A typical call costs about $0.0001–$0.001 with `gpt-6-luna` ($0.10 in / $0.50 out per million tokens). The biggest single call is the Day 5 extraction, at roughly 10,000 tokens in and 11,000 out: still under a cent. Most cells in Days 2 to 5 run **offline** and cost nothing.

## 📝 Quiz and 🏋️ exercise
- 📝 **Quiz 1** (10 questions) is on Udemy at the end of the section. Want to practise first? The questions, with an explanation for every option, are in [quizzes/week_1.yaml](../quizzes/week_1.yaml).
- 🏋️ **Exercise:** [week1_EXERCISE.ipynb](week1_EXERCISE.ipynb): add **multiple-choice questions** to the exam schema, by extending it in the notebook (no package edits!). Ten automatic checks tell you how you're doing.
- ✅ Every day's exercise has a reference solution in [solutions/](solutions/).

## 🆘 Stuck?
- Setup problems: run `uv run setup/diagnostics.py`, then see [setup/troubleshooting.ipynb](../setup/troubleshooting.ipynb)
- A red **Missing API key** panel: copy `.env.example` to `.env`, paste your `OPENROUTER_API_KEY` and **save the file**, then restart the kernel
- Lab problems: compare with [solutions/](solutions/)
- Still stuck: Udemy Q&A for the lecture you're on

⏭ **Next week:** [Week 2: RAG: building GradeWise's knowledge base](../week2/README.md)
