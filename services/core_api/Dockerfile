FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
COPY packages/research_shared/pyproject.toml packages/research_shared/
COPY services/core_api/pyproject.toml services/core_api/

COPY packages/research_shared packages/research_shared
COPY services/core_api services/core_api

RUN uv sync --package core-api --no-dev

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["core-api"]
