## 🎓 Agentic RAG with LangChain & LangGraph: Build GradeWise

# 🔄 Swapping model providers

The course uses **OpenAI** on screen, but GradeWise is **provider-agnostic**: every model is created by `init_chat_model("provider:model")`, and the names live only in your `.env` file. Swapping is a configuration change, not a code change.

> 💡 After editing `.env`, **restart your notebook kernel** so the new settings load.

---

## Option A: Anthropic (Claude)
1. Get a key at https://console.anthropic.com/ (Settings → API Keys) and add a little credit.
2. In `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   CHAT_MODEL=anthropic:claude-haiku-5-5
   SMART_MODEL=anthropic:claude-sonnet-5-5
   VISION_MODEL=anthropic:claude-sonnet-5-5
   ```
3. ⚠️ **Embeddings:** Anthropic doesn't offer an embedding model. Keep `EMBEDDING_MODEL=openai:text-embedding-3-small` (embeddings cost fractions of a cent), or use Google or local embeddings (below).

## Option B: Google Gemini (has a free tier)
1. Get a key at https://aistudio.google.com/ → **Get API key**.
2. In `.env`:
   ```
   GOOGLE_API_KEY=...
   CHAT_MODEL=google_genai:gemini-flash-latest
   SMART_MODEL=google_genai:gemini-pro-latest
   VISION_MODEL=google_genai:gemini-pro-latest
   EMBEDDING_MODEL=google_genai:gemini-embedding-001
   ```

## Option C: Free local models with Ollama
1. Install Ollama from https://ollama.com/download and start it.
2. Pull a chat model, a vision model and an embedding model (pick sizes that suit your RAM):
   ```bash
   ollama pull llama3.2           # chat (3B, about 2 GB)
   ollama pull qwen2.5vl:7b       # vision, for handwriting (about 6 GB)
   ollama pull nomic-embed-text   # embeddings
   ```
3. In `.env`:
   ```
   CHAT_MODEL=ollama:llama3.2
   SMART_MODEL=ollama:llama3.2
   VISION_MODEL=ollama:qwen2.5vl:7b
   EMBEDDING_MODEL=ollama:nomic-embed-text
   ```
4. ⚠️ **Set expectations:** small local models are slower and noticeably weaker at structured output, handwriting and grading. Everything *runs*, but your marks will disagree with a human's more often. That's a great lesson in itself (week 5's evaluation will show you exactly how much)!

## Option D: Mix and match
Each tier is independent. A popular combination: cheap OpenAI for everyday steps, Claude for grading, and local embeddings:
```
CHAT_MODEL=openai:gpt-5.4-mini
SMART_MODEL=anthropic:claude-sonnet-5-5
VISION_MODEL=openai:gpt-5.5
```

---

## 💰 Rough cost guide (whole course)
| Setup | Approximate spend |
|---|---|
| OpenAI defaults | under $10 |
| OpenAI with `BUDGET_MODE=true` | about $3 |
| Gemini free tier | $0 (rate-limited) |
| Ollama | $0 (your electricity!) |

> ⚠️ Model names change over time. If a name stops working, check the provider's model list: [OpenAI](https://platform.openai.com/docs/models) · [Anthropic](https://docs.anthropic.com/en/docs/about-claude/models) · [Gemini](https://ai.google.dev/gemini-api/docs/models) · [Ollama](https://ollama.com/library). The course CHANGELOG also announces updated defaults.

✅ **Check your swap:** `uv run python -m gradewise info` shows the models in use, and `uv run setup/diagnostics.py` confirms your keys.
