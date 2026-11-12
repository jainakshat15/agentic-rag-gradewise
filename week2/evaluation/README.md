# 📏 Week 2 Evaluation: is our retrieval actually any good?

> *"If you can't measure it, you can't improve it."* Emma is about to trust GradeWise with her students' marks, so "it seems to work" isn't good enough. This folder turns "seems to work" into numbers.

## What's here
| File | What it does |
|---|---|
| `tests.jsonl` | The test set: one CS201 question per line, with the keywords a relevant chunk must contain, a reference answer, the PDF it comes from and a category |
| `test.py` | Loads and **validates** `tests.jsonl` with Pydantic. Run it after you edit the test set |
| `eval.py` | Runs a retrieval pipeline over every question and reports the metrics below, broken down by category |

Each line of `tests.jsonl` looks like this:
```json
{"question": "What are the four necessary conditions for deadlock?",
 "keywords": ["mutual exclusion", "hold and wait", "circular wait"],
 "reference_answer": "Mutual exclusion, hold and wait, no pre-emption and circular wait.",
 "source": "unit1_operating_systems.pdf",
 "category": "direct_fact"}
```
Categories: `direct_fact` · `numerical` · `comparison` · `spanning` (the answer is spread over several chunks) · `holistic` (needs the big picture).

## The metrics
| Metric | Question it answers | Range |
|---|---|---|
| **MRR** (mean reciprocal rank) | How high up is the *first* chunk containing each keyword? Rank 1 → 1.0, rank 2 → 0.5, missing → 0 | 0–1 |
| **nDCG@k** | Are *all* the chunks containing each keyword near the top? It rewards good ordering, not just one hit | 0–1 |
| **Keywords** (coverage) | What share of the keywords appear anywhere in the top k? | 0–1 |
| **Source@k** | Did a chunk from the expected PDF make it into the top k? | 0–1 |
| **Accuracy / Completeness / Relevance** | With `--answers`: an LLM judge compares the generated answer with the reference | 1–5 |

💡 Keyword matching ignores case, Markdown bold and line-wraps, so `**Mutual**\nexclusion` still counts as "mutual exclusion".

## Run it
```bash
uv run week2/evaluation/test.py                    # validate tests.jsonl (free)
uv run week2/evaluation/eval.py                    # the pro pipeline (hybrid)
uv run week2/evaluation/eval.py --compare          # ⭐ simple vs pro, side by side, one command
uv run week2/evaluation/eval.py --pipeline dense   # or bm25, or simple
uv run week2/evaluation/eval.py --row 0            # one question in detail: what came back, and why
uv run week2/evaluation/eval.py --answers --limit 10   # also judge generated answers
uv run week2/evaluation/eval.py --offline --compare    # no API key: checks the plumbing only
```
`--compare` builds whichever vector store is missing first (the pro ingest is idempotent, so it costs nothing if nothing changed). Retrieval-only runs cost a fraction of a cent in embeddings. `--answers` calls your chat model twice per question (answer + judge), so start with `--limit 10`.

💡 **Fast & cheap run:** with `DATA_SIZE=small` in `.env`, `eval.py` scores only the first 5 questions (all from Unit 1, which the small knowledge base keeps), and the ingest scripts embed only `unit1_operating_systems.pdf` and the handbook. `--limit` and `--row` still choose for you. With `MODEL_MODE=free`, the embeddings and the judge run on OpenRouter's free models for $0; keep `DATA_SIZE=small` then, because free mode allows only about 50 requests a day.

⚠️ `--offline` uses fake embeddings in a separate folder (`output/offline_eval/`). **BM25 scores are real** (keyword search needs no embeddings), but dense, hybrid and simple scores come from random vectors and are meaningless: they only prove everything is wired up.

## 🏋️ Your turn
Open `gradewise/ingestion/chunking.py`, change `CHUNK_SIZE` / `CHUNK_OVERLAP` (or turn off `contextual_header`), re-ingest with `uv run week2/pro_implementation/ingest.py --reset`, and run `eval.py` again. Can you beat the baseline nDCG? Which categories got better, and which got worse? That trade-off is the whole game.
