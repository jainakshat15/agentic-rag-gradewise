# 🚀 Deploying GradeWise

> You've got GradeWise running on your laptop. Now let's put it on the internet, so Asha can use it from the staff room and you can drop a link in your portfolio. This guide covers three hosts, from "free and five minutes" to "production-shaped", plus LangGraph Platform as a different route altogether.

Before you start, check locally that the image builds and runs:

```bash
docker compose up --build        # then open http://localhost:8501
```

## 🔑 First, secrets (read this whichever host you pick)

- **Your keys never go in the image or in git.** `.dockerignore` excludes `.env`, and `.gitignore` excludes it too. Every host below has a "secrets" or "environment variables" screen: put `OPENAI_API_KEY` (and any model settings) there.
- **The variables GradeWise reads:** `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`), `CHAT_MODEL`, `SMART_MODEL`, `VISION_MODEL`, `EMBEDDING_MODEL`, `BUDGET_MODE`, `GRADEWISE_DEMO`, `API_HOST`, `API_PORT`, and for the UI, `API_URL`.
- **A public demo should run with `GRADEWISE_DEMO=true`.** Otherwise anyone with the link spends your money. If you deploy with real keys, put a login in front (see *Hardening* below) and set a monthly spending limit in your provider's dashboard.
- **Student work is personal data.** Real answer sheets need consent, a host in the right region, and a plan for deleting them. Day 5's "Responsible AI grading" lecture covers this; the synthetic class is the safe thing to show the world.

## 💾 What has to persist

Everything GradeWise writes goes to `/app/output` (SQLite databases, LangGraph checkpoints, the Chroma vector store, uploads and report cards). Give that folder a **persistent disk / volume**, or every redeploy starts from zero. The job registry is in memory, so run **one** API instance (more on scaling at the end).

---

## Option 1 · Hugging Face Spaces (free, the quickest public demo)

Spaces run one container on one port, which suits demo mode perfectly: run the API and the dashboard in the same container.

1. Create a Space at https://huggingface.co/new-space → **SDK: Docker** → blank template.
2. Clone it, copy in `pyproject.toml`, `uv.lock`, `README.md`, `gradewise/`, `data/` and `docker/`.
3. Add a `Dockerfile` at the Space's root that reuses ours and starts both processes (the UI on port 7860, the port Spaces expose):

   ```dockerfile
   FROM python:3.12-slim
   # ... the same lines as docker/Dockerfile, up to `USER gradewise` ...
   ENV GRADEWISE_DEMO=true API_URL=http://127.0.0.1:8000
   EXPOSE 7860
   CMD ["sh", "-c", "python -m gradewise.api & python -m streamlit run gradewise/ui/app.py --server.address=0.0.0.0 --server.port=7860"]
   ```
4. In the Space's README front matter set `sdk: docker` and `app_port: 7860`.
5. Push. The build log appears in the Space; when it says *Running*, your dashboard is live.
6. Want real models? **Settings → Variables and secrets → New secret** `OPENAI_API_KEY`, and set `GRADEWISE_DEMO=false`. ⚠️ A public Space with your key is public spending: make the Space private, or keep demo mode.

**Cost:** free CPU tier (2 vCPU, 16 GB RAM). The free tier sleeps after inactivity and its disk is wiped on restart; persistent storage is a paid add-on. Fine for a portfolio demo, not for Asha's real marks.

---

## Option 2 · Render (two services, a disk, a few dollars a month)

1. Push the repo to GitHub.
2. Render dashboard → **New → Web Service** → your repo → **Runtime: Docker**, Dockerfile path `docker/Dockerfile`.
   - Name it `gradewise-api`. Leave the start command empty (the image's `CMD` starts the API).
   - **Environment:** `API_PORT=10000` (Render routes to the port in `$PORT`, 10000 by default) and your secrets (`OPENAI_API_KEY`, ...), or `GRADEWISE_DEMO=true`.
   - **Disks → Add disk**, mount path `/app/output`, 1 GB. (Disks need a paid instance type.)
   - **Health check path:** `/health`.
3. **New → Web Service** again, same repo and Dockerfile, named `gradewise-ui`:
   - **Docker command:** `python -m streamlit run gradewise/ui/app.py --server.address=0.0.0.0 --server.port=10000`
   - **Environment:** `API_URL=https://gradewise-api.onrender.com` (the API's URL from step 2; or its private-network address `http://gradewise-api:10000`).
   - Health check path: `/_stcore/health`.
4. Open the UI's URL. 🎉

**Cost:** the free tier sleeps after 15 minutes and has no disk, so it only suits demo mode. A "Starter" instance for the API plus a 1 GB disk is roughly $7–8/month; the UI can share the same plan size. Check render.com/pricing: prices change.

> 🚀 Make it repeatable: describe both services in a `render.yaml` Blueprint and commit it. Then "deploy" is a git push.

---

## Option 3 · Fly.io (containers close to your users, volumes included)

1. Install `flyctl` and log in: `fly auth login`.
2. From the repo root: `fly launch --dockerfile docker/Dockerfile --no-deploy`. Pick a region near Asha. This writes a `fly.toml`.
3. Edit `fly.toml`: define the two processes, the internal port and the volume:

   ```toml
   [processes]
     api = "python -m gradewise.api"
     ui  = "python -m streamlit run gradewise/ui/app.py --server.address=0.0.0.0 --server.port=8501"

   [env]
     API_HOST = "0.0.0.0"
     API_URL = "http://api.process.<your-app>.internal:8000"   # the UI reaches the API over Fly's private network

   [[mounts]]
     source = "gradewise_output"
     destination = "/app/output"
     processes = ["api"]

   [http_service]
     internal_port = 8501
     processes = ["ui"]
     force_https = true
     auto_stop_machines = "stop"
     auto_start_machines = true

     [[http_service.checks]]
       path = "/_stcore/health"
       interval = "30s"
       timeout = "5s"
   ```
4. Create the volume and the secrets, then deploy:

   ```bash
   fly volumes create gradewise_output --size 1
   fly secrets set OPENAI_API_KEY=sk-...        # or: fly secrets set GRADEWISE_DEMO=true
   fly deploy
   ```
5. `fly open` opens the dashboard; `fly logs` shows both processes.

**Cost:** pay-as-you-go. Two small shared-CPU machines (1 GB RAM for the API, since Chroma and the PDF libraries like memory) plus a 1 GB volume land around $5–10/month, less if machines auto-stop when idle. See fly.io/docs/about/pricing.

---

## 💰 What it costs to RUN (the models, not the servers)

The server is the small bill. The model calls are the real one, and you already measured them in week 5. Roughly, per student with the defaults: one vision call per page, one grading call per question (two or three with tool calls), and one feedback call. For a class of 30 on a 15-question exam, that's hundreds of calls.

- `BUDGET_MODE=true` runs every step on `CHAT_MODEL`: the cheapest switch you have.
- **Grade from transcripts** when you're iterating on the grader: the vision step is usually the most expensive part.
- `consistency_runs` multiplies the grading cost: keep it at 1 unless you need the stability.
- Set a hard monthly cap in your provider's billing settings **before** you share a link.

## 🛡️ Hardening checklist (before real students' work goes in)

- [ ] A login in front: your host's access control, an OAuth proxy, or a bearer token checked by a FastAPI dependency.
- [ ] HTTPS only (all three hosts do this for you).
- [ ] Upload limits at the proxy (`server.maxUploadSize` in the UI config; a body-size limit for the API).
- [ ] Backups of `/app/output` (at least `gradewise.db`).
- [ ] LangSmith tracing on (`LANGSMITH_TRACING=true`) so you can see exactly what the grader did for any mark.
- [ ] A retention policy: delete uploads and reports at the end of term.

## 📈 When one server isn't enough

The design scales in known steps (the details are in [`gradewise/api/jobs.py`](../../gradewise/api/jobs.py)):

1. Move background work to a **task queue** (Celery, RQ, Dramatiq or Arq) with **Redis**; run grading in separate worker containers.
2. Publish progress to a **Redis Stream** per batch; the SSE endpoint reads from it, so any API replica can stream any batch.
3. Swap SQLite for **Postgres** (SQLModel works unchanged) and use `langgraph-checkpoint-postgres` for the checkpoints.
4. Swap local Chroma for a hosted vector store (Chroma Cloud, pgvector, Pinecone...).

## 🧭 The alternative: LangGraph Platform

Instead of hosting the graphs inside your own FastAPI app, you can hand them to **LangGraph Platform** (LangSmith Deployment). You describe the graphs in a `langgraph.json`:

```json
{
  "dependencies": ["."],
  "graphs": {
    "grade_batch": "gradewise.orchestrator:build_orchestrator",
    "exam": "gradewise.exam.graph:build_exam_graph"
  },
  "env": ".env"
}
```

and the platform gives you, ready-made, most of what we built by hand this week: threads and **persistent checkpoints** (Postgres), background **runs** with a task queue, **interrupts and resume** over an API, **streaming** (including our custom `emit()` events), cron jobs, and LangGraph Studio for debugging. Run it locally with `langgraph dev`, then deploy from GitHub in LangSmith.

**Trade-offs.** You get production plumbing for free and scale without writing a queue. You give up some control (the API shape is LangGraph's, not yours), add a vendor, and pay per node executed or per deployment beyond the free developer tier. A good pattern: keep the Streamlit dashboard (or a React app) and point it at the platform's API with the `langgraph-sdk` client instead of `gradewise/api`. One small change first: the platform supplies its own checkpointer, so it wants a graph compiled *without* one, while `build_orchestrator()` always attaches SQLite. A zero-argument factory that compiles the same `StateGraph` with no checkpointer is all it takes (✋ a nice exercise).

---

🎓 Shipped it? Share the link in [`community_contributions/`](../../community_contributions/README.md). Asha would be proud.
