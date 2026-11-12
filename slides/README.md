# 🎞️ Slides

The PowerPoint decks behind every video: **one deck per week**, built from one design system, so each video's slides match the notebook it teaches.

| File | What it is |
|---|---|
| `source/master_template.pptx` | Every layout once, with placeholder text and notes on when to use each one |
| `source/week_1.pptx` | Week 1: Foundations & your first win (17 videos, 280 slides) |
| `source/week_2.pptx` | Week 2: RAG, GradeWise's knowledge base (15 videos, 255 slides) |
| `source/week_3.pptx` | Week 3: Agents & LangGraph (12 videos, 262 slides) |
| `source/week_4.pptx` | Week 4: Agentic RAG & reading handwriting (12 videos, 211 slides) |
| `source/week_5.pptx` | Week 5: The grading agent, report cards & evaluation (13 videos, 245 slides) |
| `source/week_6.pptx` | Week 6: Ship it: app, Docker, deployment & the course wrap-up (14 videos, 267 slides) |
| `source/images/` | The concept diagrams the decks use (tokens, temperature, RAG pipeline, embeddings, chunking, HNSW, BM25, RRF, MMR, MRR/nDCG, architecture states `arch_w1..6`…) |
| `pdf/week_N.pdf` | The slides as PDF (1920×1080, searchable text). 40–50 MB each, so they're **not in git**: download them from the Udemy lecture resources or the GitHub Release, or export them yourself (below) |

Weeks 3 to 6 follow the same template.

## How a deck maps to the videos

Every video is one contiguous run of slides, following the course's video template:

```
TITLE → RECAP (last video's summary) → YOU ARE HERE → OBJECTIVES → content → PITFALLS → SUMMARY → NEXT
```

The first video of a week adds the **week opener** and the **6-week roadmap** (Morph: last week ticks, this week glows). The last video adds the **week-in-one-table summary**, the **quiz call-to-action** and the roadmap again (this week ticks ✓).

Every slide has **speaker notes**: the talking points for that moment of the video. In PowerPoint, View ▸ Notes. A `DEMO` slide ("Switch to the notebook") marks where the video cuts to the screen recording. It shows the notebook and the section to open.

### Week 1: `source/week_1.pptx`

| # | Lecture (video) | Slides | Notebook / guide |
|:-:|---|:-:|---|
| 1 | Day 1 - See It First: GradeWise Grades a Handwritten Exam | 1–18 | `week1/day1.ipynb` |
| 2 | Day 1 - The 6-Week Roadmap and How Every Lecture Works | 19–35 | `week1/day1.ipynb` |
| 3 | Day 1 - Setup Overview: uv, API Keys, Costs and the GitHub Repo | 36–48 | `setup/SETUP-PC.md` |
| 4 | Day 1 - Windows Setup: Git, Cursor/VS Code, Clone and `uv sync` | 49–64 | `setup/SETUP-PC.md` |
| 5 | Day 1 - Mac Setup: Git, Cursor/VS Code, Clone and `uv sync` | 65–79 | `setup/SETUP-mac.md` |
| 6 | Day 1 - Your OpenRouter Key, the `.env` File and Running Diagnostics | 80–96 | `week1/day1.ipynb` |
| 7 | Day 2 - How LLMs Work: Tokens, Context Windows, Temperature and Cost | 97–115 | `week1/day2.ipynb` |
| 8 | Day 2 - The LangChain 1.x Ecosystem: Core, LangGraph, LangSmith, Deep Agents | 116–133 | `week1/day2.ipynb` |
| 9 | Day 2 - First Calls with `init_chat_model`: Invoke, Stream, Batch, Swap Providers | 134–148 | `week1/day2.ipynb` |
| 10 | Day 3 - Prompting for Extraction: System Prompts, Templates, Examples | 149–164 | `week1/day3.ipynb` |
| 11 | Day 3 - Structured Output with Pydantic: Reliable JSON Every Time | 165–178 | `week1/day3.ipynb` |
| 12 | Day 3 - Validation, Retries and Self-Correcting Extraction | 179–196 | `week1/day3.ipynb` |
| 13 | Day 4 - Multimodal: Sending Images and PDF Pages to a Model | 197–214 | `week1/day4.ipynb` |
| 14 | Day 4 - Tools and Tool Calling: `@tool`, `bind_tools` and the Tool Loop | 215–231 | `week1/day4.ipynb` |
| 15 | 🧱 Day 5 - GradeWise Architecture Walkthrough: What We'll Build | 232–248 | `week1/day5.ipynb` |
| 16 | 🧱 Day 5 - Designing the Exam Schema, Then Extracting the Paper to JSON | 249–267 | `week1/day5.ipynb` |
| 17 | Day 5 - Week 1 Wrap-Up: Summary, Roadmap and Quiz | 268–280 | `week1/day5.ipynb` |

### Week 2: `source/week_2.pptx`

| # | Lecture (video) | Slides | Notebook |
|:-:|---|:-:|---|
| 1 | Day 1 - Week 2 Kickoff: Recap and Roadmap | 1–19 | `week2/day1.ipynb` |
| 2 | Day 1 - Why RAG? Hallucinations, Knowledge Cut-offs and Grounding | 20–37 | `week2/day1.ipynb` |
| 3 | ⭐ Day 1 - Embeddings Visually Explained: Meaning as Geometry | 38–54 | `week2/day1.ipynb` |
| 4 | Day 2 - Chunking Strategies: Fixed, Recursive, Semantic, Structure-Aware | 55–73 | `week2/day2.ipynb` |
| 5 | Day 2 - Loading PDFs: PyPDF vs PyMuPDF vs pymupdf4llm vs Docling | 74–87 | `week2/day2.ipynb` |
| 6 | Day 2 - Structure-Aware Chunking with Rich Metadata | 88–102 | `week2/day2.ipynb` |
| 7 | Day 3 - Vector Databases: HNSW, Distance Metrics and Metadata Filters | 103–120 | `week2/day3.ipynb` |
| 8 | Day 3 - A Persistent Chroma Store with Collections per Course | 121–135 | `week2/day3.ipynb` |
| 9 | Day 3 - Idempotent Ingestion: Hashing, De-duplication, Re-indexing | 136–153 | `week2/day3.ipynb` |
| 10 | Day 4 - Retrieval Strategies: Dense, BM25, Hybrid, MMR, Multi-Query, Reranking | 154–172 | `week2/day4.ipynb` |
| 11 | Day 4 - Hybrid Search and Reranking in Practice | 173–187 | `week2/day4.ipynb` |
| 12 | Day 4 - Your First RAG Chain: Answers with Citations | 188–203 | `week2/day4.ipynb` |
| 13 | 🧱 Day 5 - Measuring Retrieval: MRR, nDCG and Recall@k with `tests.jsonl` | 204–222 | `week2/day5.ipynb` |
| 14 | 🧱 Day 5 - From Simple to Pro: The GradeWise Ingestion Pipeline and CLI | 223–241 | `week2/day5.ipynb` |
| 15 | Day 5 - Week 2 Wrap-Up: Summary, Roadmap and Quiz | 242–255 | `week2/day5.ipynb` |

### Week 3: `source/week_3.pptx`

Graph diagrams are native shapes that **Morph** step by step: state flowing through the nodes, the ReAct loop, the `Send` fan-out, the interrupt and resume.

| # | Lecture (video) | Slides | Notebook |
|:-:|---|:-:|---|
| 1 | Day 1 - Week 3 Kickoff: Recap and Roadmap | 1–21 | `week3/day1.ipynb` |
| 2 | Day 1 - What Is an Agent? Workflows vs Agents and the Core Patterns | 22–43 | `week3/day1.ipynb` |
| 3 | Day 1 - The ReAct Loop and Designing Great Tools | 44–66 | `week3/day1.ipynb` |
| 4 | Day 2 - `create_agent`: Tools, Memory and Structured Responses | 67–86 | `week3/day2.ipynb` |
| 5 | Day 2 - Middleware: Logging, Summarization, Retries, Limits, PII, Human Approval | 87–107 | `week3/day2.ipynb` |
| 6 | Day 3 - LangGraph Concepts: State, Nodes, Edges, Reducers, Supersteps | 108–132 | `week3/day3.ipynb` |
| 7 | Day 3 - Your First StateGraph with Conditional Routing and Loops | 133–156 | `week3/day3.ipynb` |
| 8 | Day 4 - Parallel Map-Reduce with `Send` | 157–176 | `week3/day4.ipynb` |
| 9 | Day 4 - Checkpointers and Human-in-the-Loop with `interrupt()` | 177–198 | `week3/day4.ipynb` |
| 10 | Day 4 - Subgraphs, Streaming and LangGraph Studio | 199–216 | `week3/day4.ipynb` |
| 11 | 🧱 Day 5 - Exam Understanding v2: Extract → Validate → Retry → Admin Approval → Store | 217–247 | `week3/day5.ipynb` |
| 12 | Day 5 - Week 3 Wrap-Up: Summary, Roadmap and Quiz | 248–262 | `week3/day5.ipynb` |

### Week 4: `source/week_4.pptx`

Diagrams **Morph** step by step: the agentic RAG loop (route, retrieve, grade, rewrite, answer), the transcription subgraph, the page fan-out with real page thumbnails, and double-pass verification. The handwriting slides use real pages from `data/answer-sheets/synthetic/`.

| # | Lecture (video) | Slides | Notebook |
|:-:|---|:-:|---|
| 1 | Day 1 - Week 4 Kickoff: Recap and Roadmap | 1–19 | `week4/day1.ipynb` |
| 2 | Day 1 - The Agentic RAG Spectrum: Adaptive, Corrective and Self-RAG | 20–38 | `week4/day1.ipynb` |
| 3 | Day 1 - Building the Agentic RAG Graph: Retrieve, Grade, Rewrite, Answer | 39–60 | `week4/day1.ipynb` |
| 4 | Day 2 - Grounded Answers: Citations and Hallucination Checks | 61–76 | `week4/day2.ipynb` |
| 5 | 🧱 Day 2 - The GradeWise Evidence Retriever for Grading | 77–92 | `week4/day2.ipynb` |
| 6 | Day 3 - OCR vs HTR vs Vision LLMs: Choosing a Handwriting Strategy | 93–106 | `week4/day3.ipynb` |
| 7 | Day 3 - PDF Pages to Images: DPI, Rotation, Contrast and Pre-Processing | 107–123 | `week4/day3.ipynb` |
| 8 | Day 4 - Faithful Transcription with a Vision LLM (Never "Fix" the Student) | 124–138 | `week4/day4.ipynb` |
| 9 | Day 4 - Mapping Answers to Questions: Labels, Continuations, Missing Answers | 139–154 | `week4/day4.ipynb` |
| 10 | Day 4 - Confidence Scores, Illegible Flags and Double-Pass Verification | 155–173 | `week4/day4.ipynb` |
| 11 | 🧱 Day 5 - The Transcription Subgraph, Tuned for Cost and Latency | 174–197 | `week4/day5.ipynb` |
| 12 | Day 5 - Week 4 Wrap-Up: Summary, Roadmap and Quiz | 198–211 | `week4/day5.ipynb` |

### Week 5: `source/week_5.pptx`

Four native build-ups **Morph** step by step: the `Send` grading fan-out, the orchestrator graph node by node, interrupt → checkpoint → resume tomorrow, and the MAE vs QWK worked example. Real visuals: the HTML/PDF report card, the week 6 app's event log, review queue and chat, and student 10's actual prompt-injection line.

| # | Lecture (video) | Slides | Notebook |
|:-:|---|:-:|---|
| 1 | Day 1 - Week 5 Kickoff: Recap and Roadmap | 1–19 | `week5/day1.ipynb` |
| 2 | Day 1 - LLM-as-Grader: Rubrics, Bias, Consistency and Human Oversight | 20–35 | `week5/day1.ipynb` |
| 3 | ⭐ Day 1 - Grade One Answer: Rubric + Retrieved Evidence → `GradeResult` | 36–58 | `week5/day1.ipynb` |
| 4 | Day 2 - Grading Tools: `retrieve_reference`, `get_rubric`, `flag_for_review` | 59–76 | `week5/day2.ipynb` |
| 5 | Day 2 - Grading Every Question in Parallel with `Send` | 77–96 | `week5/day2.ipynb` |
| 6 | Day 2 - Reliable Marks: Guardrails, Self-Consistency and Prompt-Injection Defense | 97–115 | `week5/day2.ipynb` |
| 7 | Day 3 - Writing the Report Card: Scores, Feedback, Citations (HTML/PDF) | 116–131 | `week5/day3.ipynb` |
| 8 | Day 3 - Storing Reports and the "Chat With Results" Agent | 132–147 | `week5/day3.ipynb` |
| 9 | 🧱 Day 4 - The Orchestrator: GradeWise End to End | 148–175 | `week5/day4.ipynb` |
| 10 | Day 4 - LangSmith: Tracing, Costs and Debugging a Full Run | 176–190 | `week5/day4.ipynb` |
| 11 | 🧱 Day 5 - Grader vs Human: Golden Set, MAE and Quadratic Weighted Kappa | 191–215 | `week5/day5.ipynb` |
| 12 | Day 5 - Transcription CER/WER and Regression Tests with pytest | 216–230 | `week5/day5.ipynb` |
| 13 | Day 5 - Week 5 Wrap-Up: Summary, Roadmap and Quiz | 231–245 | `week5/day5.ipynb` |

### Week 6: `source/week_6.pptx`

Native build-ups **Morph** step by step: the production architecture (UI → API → workers → storage → the event stream back), request → `202` → background job → SSE stream → `event: end` → review and reconnect, the Dockerfile layer by layer (plus what a code edit vs a new package rebuilds), the four deploy options, the portfolio checklist, and the **final roadmap**: one slide per week's milestone, ticking all six weeks. Real visuals: every dashboard page (home, knowledge base, exams, grade a batch, the review queue, reports, the report card, chat), the compose diagram and the ten handbook principles. The course closes with a thank-you slide and the final-quiz call to action (Quiz 6 + the 20-question final quiz).

| # | Lecture (video) | Slides | Notebook |
|:-:|---|:-:|---|
| 1 | Day 1 - Week 6 Kickoff: Recap and Roadmap | 1–20 | `week6/day1.ipynb` |
| 2 | Day 1 - Production Architecture: API, Workers, UI, Storage | 21–42 | `week6/day1.ipynb` |
| 3 | Day 1 - FastAPI Backend: Upload, Run and Status Endpoints | 43–59 | `week6/day1.ipynb` |
| 4 | Day 2 - Streamlit Admin Dashboard: Knowledge Bases, Exams, Batches | 60–76 | `week6/day2.ipynb` |
| 5 | Day 2 - The Review Queue: Approving Flags and Overriding Marks | 77–97 | `week6/day2.ipynb` |
| 6 | Day 3 - Streaming Grading Progress from LangGraph to the UI | 98–119 | `week6/day3.ipynb` |
| 7 | Day 3 - Report Card Viewer and Chat With Results | 120–135 | `week6/day3.ipynb` |
| 8 | Day 4 - One Command to Run It All: Docker Compose | 136–160 | `week6/day4.ipynb` |
| 9 | Day 4 - Deploying GradeWise (Render / Fly.io / HF Spaces) | 161–180 | `week6/day4.ipynb` |
| 10 | Day 5 - Responsible AI Grading: Fairness, Transparency, Privacy | 181–200 | `week6/day5.ipynb` |
| 11 | Day 5 - Capstone Showcase: From Upload to Report Card | 201–219 | `week6/day5.ipynb` |
| 12 | Day 5 - Where Next: Math and Diagrams, Deep Agents, an MCP Server for GradeWise | 220–232 | `week6/day5.ipynb` |
| 13 | Day 5 - Make It Your Portfolio Piece | 233–246 | `week6/day5.ipynb` |
| 14 | Day 5 - The Final Roadmap and Thank You | 247–267 | `week6/day5.ipynb` |

## The design system

- **16:9, dark, on a grid.** 1920 × 1080 with a 12-column grid: 96 px outer margins, 24 px gutters, an 8 px spacing scale. Every slide title sits at the same place, so slides line up and Morph cleanly.
- **Colour.** Background `#0B1020`, surfaces `#141B34` / `#1B2445` with a 1 px `#232C52` hairline, primary `#7C5CFF`, accent `#22D3EE`, success / warn / error `#34D399 / #FBBF24 / #F87171`, text `#E6E9F5`, muted `#9AA3C7`. Each week has its own accent: the same colours as the banners and roadmap in `assets/`.
- **Type.** **Inter Display** SemiBold for headlines (slightly tightened), **Inter** Regular / Medium / SemiBold for text, **JetBrains Mono** for code and file paths. All three are free (SIL Open Font License). One size scale: 50 / 44 / 32 pt headlines, 24 / 20 pt card titles, 22 / 18 / 15 pt text, 12 pt labels; nothing below 10.5 pt (21 px). Labels are sentence case, never shouty caps, and emoji are stripped from slide text.
- **Layouts:**

  | | | | | | |
  |---|---|---|---|---|---|
  | TITLE | WEEK-OPENER | RECAP | ROADMAP-FULL | ROADMAP-MINI | OBJECTIVES |
  | CONCEPT | DIAGRAM | FLOW | ANALOGY | CODE | DEMO |
  | COMPARE | PITFALL | GRADEWISE-LINK | SUMMARY | WEEK-SUMMARY | QUIZ-CTA |
  | NEXT | | | | | |

  See `source/master_template.pptx` for one of each. Weeks 3 to 6 add their own build-up layouts (graphs, pipelines, the final quiz and the thank-you) in the same style.
- **On every slide but the title:** a quiet footer with the GradeWise mark, *Week · Day · Video*, and the course progress bar (day *k* of 30).
- **The avatar safe zone.** The presenter is composited bottom-right in the videos, so the area right of x = 1600 px and below y = 760 px is always empty: no text, shapes, logos or progress elements (only the background runs into it). The build checks every shape on every slide and fails if anything lands there.
- **Transitions:**
  - **Morph** where a diagram builds up across slides (shapes share `!!name` names) and on the roadmap. This needs PowerPoint 2019 or Microsoft 365; older versions fall back to Fade.
  - **Fade** everywhere else.
- Shapes are native PowerPoint shapes (editable). Concept diagrams are PNGs in `source/images/`, drawn in the same palette and fonts.

## Editing the decks

The decks are generated from data by the course's build tooling, so every slide follows the same design system. You're welcome to edit your own copy in any presentation app; just know that the next course update replaces these files.

## 📄 Exporting the PDFs yourself

Any presentation app can open the `.pptx` files. To export with [DeckCraft](https://getartcraft.com/apps/deckcraft)'s command line (it keeps a searchable text layer):

```bash
deckcraft-cli run --in slides/source/week_1.pptx --cmd 'file.export={"path":"slides/pdf/week_1.pdf","format":"pdf","dpi":144,"textLayer":true}'
```

In PowerPoint or Keynote, use File ▸ Export ▸ PDF. The decks use **Inter**, **Inter Display** and **JetBrains Mono** (all free, from [rsms/inter](https://github.com/rsms/inter) and [JetBrains/JetBrainsMono](https://github.com/JetBrains/JetBrainsMono)). Install them first, or the slides fall back to your system fonts.
