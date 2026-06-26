FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
COPY packages/research_shared/pyproject.toml packages/research_shared/
COPY services/core_api/pyproject.toml services/core_api/
COPY services/worker/pyproject.toml services/worker/
COPY services/vk_bot/pyproject.toml services/vk_bot/

COPY packages/research_shared packages/research_shared
COPY services/core_api services/core_api

RUN uv sync --package core-api --no-dev

ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN mkdir -p "${FASTEMBED_CACHE_PATH}" && /app/.venv/bin/python -c "\
from fastembed import SparseTextEmbedding, TextEmbedding; \
TextEmbedding(model_name='intfloat/multilingual-e5-large'); \
SparseTextEmbedding(model_name='Qdrant/bm25')"

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app
COPY --from=builder /app/.cache/fastembed /app/.cache/fastembed

ENV PATH="/app/.venv/bin:$PATH"
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed

EXPOSE 8000

CMD ["core-api"]
