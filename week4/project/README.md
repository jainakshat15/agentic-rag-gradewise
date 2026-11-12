# Week 4 project: the GradeWise milestone code

<!-- Other week 4 demos (agentic RAG, evidence retriever) have their own sections in this file. -->

## Transcription demo

> **Emma has a stack of 30 handwritten scripts. Today GradeWise reads them for her, word for word.**

`transcription_graph.py` runs the week 4 transcription subgraph on ONE handwritten answer sheet and shows you, question by question, what the vision model read and how sure it is.

```bash
uv run week4/project/transcription_graph.py data/answer-sheets/synthetic/student_03.pdf
```

### What happens under the hood

```
render ─▶ transcribe_page ×N (Send fan-out, in parallel) ─▶ map ─▶ verify_low_confidence ─▶ END
```

| Step | Code | What it does |
|---|---|---|
| 🖼️ render | `gradewise/transcription/pdf_to_images.py` | PDF pages → images at 200 DPI, then grayscale, auto-contrast and a size cap (pixels cost tokens!) |
| 👁️ transcribe_page | `gradewise/transcription/vision.py` | One multimodal message per page (text + image content blocks) → a structured `PageTranscription`. **Verbatim**: Sam's "semafore" stays "semafore" |
| 🧭 map | `gradewise/transcription/mapper.py` | Pure Python: labels like `4(b)` / `Answer 4 b` → `Q4b`, "contd." across pages, out-of-order answers, missing answers reported as blank (never invented) |
| 🔁 verify_low_confidence | `gradewise/transcription/confidence.py` | Pages behind low-confidence answers are read a second time at 300 DPI. If the two readings disagree (CER > 15%), the answer is flagged for Emma to check |

### What you will see
- A cost panel: pages, vision calls, tokens and a **rough** cost estimate (priced with the `ROUGH_PRICE_PER_M_*` constants in `gradewise/transcription/graph.py`).
- A table: question · pages · confidence (red = needs review, below 0.75) · the first 60 characters.
- If `transcripts.json` sits next to the PDF (it does for the synthetic sheets), an extra **CER** column: the character error rate against what the student really wrote.
- The full `AnswerSheet` saved to `output/transcriptions/<student>.json`, ready for week 5's grader.

### Options
```bash
uv run week4/project/transcription_graph.py --help
uv run week4/project/transcription_graph.py data/answer-sheets/synthetic/student_07.pdf --dpi 150
uv run week4/project/transcription_graph.py data/answer-sheets/synthetic/student_07.pdf --no-verify
```

🧪 **TRY:** run the same sheet at `--dpi 150`, `200` and `300`. Does the CER improve enough to justify the extra tokens?

### 💰 What a run costs
- `MODEL_MODE=cheap` (the default): the vision model is `openai/gpt-6-luna` via OpenRouter ($0.10 in / $0.50 out per 1M tokens), so one sheet costs a fraction of a cent.
- `MODEL_MODE=free`: `openrouter/free`, OpenRouter's router over free models (same `OPENROUTER_API_KEY`, no credits), $0. Expect it to be slower: 20 requests a minute and 50 a day (1,000 once you've ever bought $10 of credits), so GradeWise retries 429 errors and reads at most 2 pages at once. A different free model answers each call, so CER varies, and free providers may log or train on prompts: synthetic sheets only, never real students' scripts.
- 📦 `DATA_SIZE=small`: with no PDF argument, the demo reads `student_01` instead of `student_03`.

⚠️ **No API key?** You'll get a friendly *Missing API key* panel telling you how to fix your `.env`. The tests (`uv run python -m pytest tests/test_transcription.py -q`) use a fake vision model, so they run for free without one.
