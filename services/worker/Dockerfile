FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
COPY packages/research_shared/pyproject.toml packages/research_shared/
COPY services/worker/pyproject.toml services/worker/
COPY services/core_api/pyproject.toml services/core_api/
COPY services/vk_bot/pyproject.toml services/vk_bot/

COPY packages/research_shared packages/research_shared
COPY services/worker services/worker

RUN uv sync --package worker --no-dev

ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN mkdir -p "${FASTEMBED_CACHE_PATH}" && /app/.venv/bin/python -c "\
from fastembed import SparseTextEmbedding, TextEmbedding; \
TextEmbedding(model_name='intfloat/multilingual-e5-large'); \
SparseTextEmbedding(model_name='Qdrant/bm25')"

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app /app
COPY --from=builder /app/.cache/fastembed /app/.cache/fastembed

ENV PATH="/app/.venv/bin:$PATH"
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed

CMD ["celery", "-A", "worker.celery_app", "worker", "--loglevel=info"]
