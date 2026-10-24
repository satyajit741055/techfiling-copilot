# =============================================================================
# Backend Dockerfile - TechFiling Copilot (FastAPI + RAG pipeline)
# =============================================================================
# The container runs in Hugging Face API mode only (see docker-compose.yml), so
# the local-inference stack (torch, sentence-transformers, transformers) is not
# installed. Those packages are only imported lazily by the "local" backends.
# =============================================================================

FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
        --no-install-package torch \
        --no-install-package sentence-transformers \
        --no-install-package transformers

# =============================================================================
# Runtime stage
# =============================================================================
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 1000 app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --chown=app:app api/ ./api/
COPY --chown=app:app src/ ./src/
COPY --chown=app:app data/ ./data/

RUN mkdir -p src/logs && chown app:app src/logs

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
