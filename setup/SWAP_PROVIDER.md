## 🎓 Agentic RAG with LangChain & LangGraph: Build GradeWise

# 🔄 Choosing your models: `MODEL_MODE`

GradeWise is **provider-agnostic**: every model is created by `init_chat_model("provider:model")`, and the names live only in your `.env` file. The course runs on **[OpenRouter](https://openrouter.ai)**, one API that reaches hundreds of models, so **one key, `OPENROUTER_API_KEY`**, covers everything. One more line picks the models:

```
OPENROUTER_API_KEY=sk-or-v1-...
MODEL_MODE=cheap     # cheap (the default) | free | custom
```

| `MODEL_MODE` | Key | Chat · grading · vision | Embeddings | Cost |
|---|---|---|---|---|
| `cheap` (**default**) | `OPENROUTER_API_KEY` | `openrouter:openai/gpt-6-luna` (grading thinks harder) | `openrouter:openai/text-embedding-3-small` | $0.10 / $0.50 per 1M tokens in/out; embeddings $0.02 per 1M |
| `free` | `OPENROUTER_API_KEY` (no credits needed) | `openrouter:openrouter/free` | `openrouter:nvidia/nemotron-3.5-embed-1b:free` | $0 |
| `custom` | whichever provider you use | whatever `CHAT_MODEL` / `SMART_MODEL` / `VISION_MODEL` say | `EMBEDDING_MODEL` | yours |

> 💡 After editing `.env`, **restart your notebook kernel** so the new settings load.
>
> 💡 `cheap` and `free` **ignore** the four `*_MODEL` lines, so switching really is one line. Only `custom` reads them.

✅ **Check it:** `uv run python -m gradewise info` shows the mode and models in use, and `uv run setup/diagnostics.py` tests your OpenRouter key (a free call that uses no tokens).

---

## 🔑 Getting your OpenRouter key
1. Sign up at https://openrouter.ai (a Google or GitHub login works).
2. Go to https://openrouter.ai/keys → **Create Key**. Name it `gradewise-course` and copy it right away: it starts with `sk-or-`.
3. **For `cheap` mode, add credits:** https://openrouter.ai/settings/credits. A few dollars is plenty for the course; OpenRouter adds a small fee when you buy credits. **`free` mode needs no credits.**
4. Optional, but wise: set a credit **limit** on the key (same page as step 2) so there are never surprises.
5. Paste it into `.env` as `OPENROUTER_API_KEY=sk-or-v1-...` (no quotes, no spaces) and **save**.

---

## `cheap`: gpt-6-luna (the default)
```
MODEL_MODE=cheap
```
All three tiers use **openai/gpt-6-luna**: CHAT and VISION at a low reasoning effort, SMART (grading) at a higher one. OpenRouter passes OpenAI's price straight through: $0.10 in / $0.50 out per 1M tokens. Week 1 costs well under a cent per call (about $0.0001–$0.001), a few cents for the whole week, and GradeWise prints the cost of each run so you always know.

## `free`: OpenRouter's free models ($0)
```
MODEL_MODE=free
DATA_SIZE=small     # strongly recommended with free mode
```
`openrouter/free` is OpenRouter's **free-models router**: for each request it keeps only the free models that support what the request needs (images, tool calling, structured output) and picks one at random. The reply says which model answered (GradeWise shows it in its logs). Embeddings use **nvidia/nemotron-3.5-embed-1b:free**.

> ⚠️ **Free-mode caveats, read before you switch:**
> 1. **Rate limits:** 20 requests a minute and **50 requests a day** on an account that has bought less than $10 of credits ever; 1,000 a day once you've ever bought $10 or more.
> 2. **Use it with `DATA_SIZE=small`:** a full class run makes hundreds of calls. GradeWise prints a warning if you combine `free` with `DATA_SIZE=full`.
> 3. **Quality and data:** the model changes from call to call, so results vary, and an occasional structured-output failure is normal (GradeWise asks again). Free providers may log prompts or train on them: fine for this course's fictional data, **never OK for real student data**. You may need to allow it in your [OpenRouter privacy settings](https://openrouter.ai/settings/privacy): enable **"free endpoints that may train on inputs"**, or the router finds no model.
>
> In `free` mode GradeWise retries 429 ("too many requests") errors with backoff and lowers parallelism (`max_concurrency` 2), so runs are slower but finish. When the **daily** limit is used up it stops with a clear message instead of retrying.

## `custom`: pick every model yourself
Set `MODEL_MODE=custom`, then each tier is independent. The defaults are the cheap models, so you only write the lines you change:
```
MODEL_MODE=custom
CHAT_MODEL=openrouter:openai/gpt-6-luna
SMART_MODEL=openrouter:openai/gpt-6-luna
VISION_MODEL=openrouter:openai/gpt-6-luna
EMBEDDING_MODEL=openrouter:openai/text-embedding-3-small
```
Any model on https://openrouter.ai/models works with the same key: write it as `openrouter:<vendor>/<model>`.

### 🚀 Optional upgrade: a stronger grader
Grading is the step where model quality matters most. For a stronger marker, keep everything cheap and upgrade only the SMART tier:
```
MODEL_MODE=custom
SMART_MODEL=openrouter:openai/gpt-6.1-terra     # $1 in / $5 out per 1M tokens
```
Week 5's evaluation shows you whether the extra cost buys closer agreement with the human examiner.

### Calling a provider directly (its own key)
Prefer to use a provider's own API? Add its key to the "custom only" section of `.env`:

| Provider | `.env` | Example models |
|---|---|---|
| OpenAI | `OPENAI_API_KEY=sk-...` | `openai:gpt-6-luna`, `EMBEDDING_MODEL=openai:text-embedding-3-small` |
| Anthropic (Claude) | `ANTHROPIC_API_KEY=sk-ant-...` | `anthropic:claude-haiku-5-5`, `anthropic:claude-sonnet-5-5` (no embeddings: keep an OpenRouter or OpenAI one) |
| Google Gemini | `GOOGLE_API_KEY=AIza...` (https://aistudio.google.com/apikey) | `google_genai:gemini-3.1-flash-lite`, `EMBEDDING_MODEL=google_genai:gemini-embedding-001` |

### Free local models with Ollama
1. Install Ollama from https://ollama.com/download and start it.
2. Pull a chat model, a vision model and an embedding model (pick sizes that suit your RAM):
   ```bash
   ollama pull llama3.2           # chat (3B, about 2 GB)
   ollama pull qwen2.5vl:7b       # vision, for handwriting (about 6 GB)
   ollama pull nomic-embed-text   # embeddings
   ```
3. In `.env`:
   ```
   MODEL_MODE=custom
   CHAT_MODEL=ollama:llama3.2
   SMART_MODEL=ollama:llama3.2
   VISION_MODEL=ollama:qwen2.5vl:7b
   EMBEDDING_MODEL=ollama:nomic-embed-text
   ```
4. ⚠️ **Set expectations:** small local models are slower and noticeably weaker at structured output, handwriting and grading. Everything *runs*, but your marks will disagree with a human's more often. That's a great lesson in itself (week 5's evaluation shows you exactly how much)!

---

## 🔢 Switching modes and your vector store
Different embedding models produce vectors of different sizes, which can't be mixed. So GradeWise keeps **one vector store per embedding model**: `output/vector_db/openrouter_openai_text-embedding-3-small/`, `output/vector_db/openrouter_nvidia_nemotron-3.5-embed-1b_free/`, and so on. Switching mode never breaks your store; the first run in a new mode simply ingests the notes again ($0 in `free`, a fraction of a cent in `cheap`).

## 📦 Less data: `DATA_SIZE=small`
Independent of the model mode, `DATA_SIZE=small` runs everything on a slice: 2 knowledge-base PDFs (`unit1_operating_systems.pdf` + `cs201_course_handbook.pdf`), exam questions Q1–Q3, and 2 students (`student_01` and `student_05`). It's for when you know the ideas and just want to run the APIs quickly and cheaply, and it's the way to use `free` mode.

## 💰 Prices used in the cost readouts (verified 2026-11-12)
| Model (via OpenRouter) | Input | Cached input | Output |
|---|---|---|---|
| `openai/gpt-6-luna` | $0.10 / 1M | $0.01 / 1M | $0.50 / 1M |
| `openai/gpt-6.1-terra` | $1 / 1M | | $5 / 1M |
| `openai/text-embedding-3-small` | $0.02 / 1M | | |
| `openrouter/free`, any `:free` model | $0 | | $0 |

OpenRouter passes the provider's price through; sources: [openrouter.ai/models](https://openrouter.ai/models) and the [OpenRouter docs](https://openrouter.ai/docs).

> ⚠️ Model names and prices change over time. If a name stops working, check https://openrouter.ai/models (or your provider's model list in custom mode). The course CHANGELOG also announces updated defaults.
