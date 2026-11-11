# ╭──────────────────────────────────────────────────────────────────────╮
# │  🎓 Agentic RAG with LangChain & LangGraph                           │
# │  🐳 Week 6 · Day 4: docker/Dockerfile, one image for API and UI      │
# │                                                                      │
# │  What this file does                                                 │
# │    Builds a small, reproducible Linux image with GradeWise and its   │
# │    exact locked dependencies (uv.lock). The same image runs both     │
# │    services; docker-compose.yml just starts it with two commands:    │
# │      api → python -m gradewise.api        (port 8000, the default)   │
# │      ui  → python -m streamlit run gradewise/ui/app.py  (port 8501)  │
# │                                                                      │
# │  Build : docker build -f docker/Dockerfile -t gradewise .            │
# │  Run   : docker run --rm -p 8000:8000 -e GRADEWISE_DEMO=true gradewise│
# │  ⏭ Used by: docker-compose.yml, week6/project/DEPLOY.md             │
# ╰──────────────────────────────────────────────────────────────────────╯

FROM python:3.12-slim

# 💡 uv as a single static binary, copied from its official image. Pin the version: builds must be repeatable.
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /app

# 🧱 Layer 1: dependencies only. This layer is cached until pyproject.toml or uv.lock change, so editing
#    your code rebuilds in seconds instead of re-downloading LangChain, Chroma and friends every time.
#    --frozen  : install EXACTLY uv.lock; fail instead of silently re-resolving.
#    --no-dev  : skip pytest, ruff, nbconvert (not needed to serve).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 🧱 Layer 2: the code and the course data GradeWise reads (exam pack, course notes, synthetic class).
COPY gradewise ./gradewise
COPY data ./data
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# 🔒 Never run as root: if anything is ever exploited, it owns one unprivileged user, not the container.
RUN useradd --create-home --uid 1000 gradewise \
    && mkdir -p /app/output \
    && chown -R gradewise:gradewise /app/output
USER gradewise

# 💡 output/ (databases, vector store, report cards, uploads) is a VOLUME: it outlives the container.
VOLUME ["/app/output"]
EXPOSE 8000 8501

# 🩺 Docker (and Compose, Fly, Render...) call this to know the API is really serving, not just started.
#    python instead of curl: the slim image has no curl, and adding it costs megabytes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"API_PORT\",\"8000\")}/health', timeout=4)" || exit 1

# ⚠️ ONE worker: the batch registry lives in memory (see gradewise/api/jobs.py for the production path).
CMD ["python", "-m", "gradewise.api"]
