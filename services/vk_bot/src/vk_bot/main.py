from __future__ import annotations

import asyncio
import logging
import os
import signal
import time

import httpx
import redis.asyncio as aioredis

from research_shared.logging_config import get_logger
from vk_bot.config import VkBotSettings, get_settings
from vk_bot.core_api.client import CoreApiClient
from vk_bot.handlers.bot_service import BotService
from vk_bot.handlers.router import CommandRouter
from vk_bot.security.rate_limiter import create_rate_limiter
from vk_bot.security.sanitizer import MessageSanitizer
from vk_bot.state.conversation import ConversationStore
from vk_bot.state.message_dedup import create_message_dedup_store
from vk_bot.state.upload_naming import create_upload_naming_store
from vk_bot.state.user_queue import BatchPoller, UserUploadQueueStore
from vk_bot.state.user_session import create_user_session_store
from vk_bot.logging_setup import setup_logging
from vk_bot.transport.factory import create_vk_transport
from vk_bot.vk.api import VkApiClient
from vk_bot.vk.message_enricher import MessageEnricher

logger = get_logger(__name__)


async def wait_for_core_api(
  settings: VkBotSettings,
  *,
  client: httpx.AsyncClient | None = None,
) -> None:
  owns_client = client is None
  probe_client = client or httpx.AsyncClient(
    base_url=settings.core_api_base_url.rstrip("/"),
    timeout=settings.vk_core_api_timeout_seconds,
    headers={"X-Client": "vk_bot"},
  )
  timeout_seconds = settings.vk_core_api_startup_wait_seconds
  poll_interval = settings.vk_core_api_startup_poll_interval_seconds
  started_at = time.monotonic()
  attempt = 0
  delay = poll_interval

  try:
    while True:
      attempt += 1
      elapsed_ms = int((time.monotonic() - started_at) * 1000)
      try:
        response = await probe_client.get("/health")
        if response.status_code == 200:
          logger.info(
            "Core API is ready",
            extra={
              "event": "vk_bot.core_api_ready",
              "attempts": attempt,
              "elapsed_ms": elapsed_ms,
            },
          )
          return
      except httpx.HTTPError:
        pass

      if time.monotonic() - started_at >= timeout_seconds:
        logger.error(
          "Timed out waiting for Core API",
          extra={
            "event": "vk_bot.core_api_wait_timeout",
            "attempts": attempt,
            "elapsed_ms": elapsed_ms,
            "timeout_seconds": timeout_seconds,
          },
        )
        raise SystemExit(1)

      remaining = timeout_seconds - (time.monotonic() - started_at)
      await asyncio.sleep(min(delay, max(remaining, 0)))
      delay = min(delay * 2, poll_interval * 8)
  finally:
    if owns_client:
      await probe_client.aclose()


async def _run() -> None:
  settings = get_settings()
  setup_logging(settings)
  storage_backend = os.environ.get("STORAGE_BACKEND", "local")
  logger.info(
    "Starting VK bot",
    extra={
      "event": "vk_bot.start",
      "git_sha": os.environ.get("GIT_SHA", "unknown"),
      "redirect_handler": "shared",
      "storage_backend": storage_backend,
    },
  )
  redis_client = aioredis.from_url(settings.effective_redis_url, decode_responses=True)

  vk_api = VkApiClient(settings)
  core_api = CoreApiClient(settings)
  await wait_for_core_api(settings)
  rate_limiter = create_rate_limiter(settings, redis_client)
  sanitizer = MessageSanitizer(settings)
  queue_store = UserUploadQueueStore(settings, redis_client)
  naming_store = create_upload_naming_store(settings, redis_client)
  message_enricher = MessageEnricher(vk_api, settings)

  conversation_store = ConversationStore(
    redis=redis_client,
    prefix=settings.redis_key_prefix,
    max_turns=settings.vk_conversation_history_max_turns,
    ttl_seconds=settings.vk_conversation_history_ttl_seconds,
  ) if settings.vk_conversation_history_enabled else None

  bot_service = BotService(
    settings=settings,
    vk_api=vk_api,
    core_api=core_api,
    rate_limiter=rate_limiter,
    sanitizer=sanitizer,
    queue_store=queue_store,
    router=CommandRouter(settings),
    session_store=create_user_session_store(settings, redis_client),
    naming_store=naming_store,
    dedup_store=create_message_dedup_store(settings, redis_client),
    conversation_store=conversation_store,
  )

  poller = BatchPoller(settings, queue_store, core_api, vk_api)
  await poller.start()

  transport = create_vk_transport(settings, message_enricher=message_enricher)
  stop_event = asyncio.Event()

  def _request_stop() -> None:
    stop_event.set()

  loop = asyncio.get_running_loop()
  for sig in (signal.SIGINT, signal.SIGTERM):
    try:
      loop.add_signal_handler(sig, _request_stop)
    except NotImplementedError:
      pass

  transport_task = asyncio.create_task(transport.run(bot_service.handle))
  stop_task = asyncio.create_task(stop_event.wait())

  try:
    done, _ = await asyncio.wait(
      {transport_task, stop_task},
      return_when=asyncio.FIRST_COMPLETED,
    )
    for task in done:
      if task is transport_task and transport_task.exception():
        raise transport_task.exception()  # type: ignore[misc]
  finally:
    await transport.stop()
    transport_task.cancel()
    await poller.stop()
    await core_api.aclose()
    await redis_client.aclose()


def run() -> None:
  settings = get_settings()
  setup_logging(settings)
  asyncio.run(_run())


if __name__ == "__main__":
  run()
