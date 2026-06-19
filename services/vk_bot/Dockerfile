FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
COPY packages/research_shared/pyproject.toml packages/research_shared/
COPY services/vk_bot/pyproject.toml services/vk_bot/
COPY services/core_api/pyproject.toml services/core_api/
COPY services/worker/pyproject.toml services/worker/

COPY packages/research_shared packages/research_shared
COPY services/vk_bot services/vk_bot

RUN uv sync --package vk-bot --no-dev

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8081

CMD ["vk-bot"]
