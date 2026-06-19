from __future__ import annotations

import asyncio
import logging
import signal

import redis.asyncio as aioredis

from research_shared.logging_config import get_logger
from vk_bot.config import get_settings
from vk_bot.core_api.client import CoreApiClient
from vk_bot.handlers.bot_service import BotService
from vk_bot.handlers.router import CommandRouter
from vk_bot.security.rate_limiter import create_rate_limiter
from vk_bot.security.sanitizer import MessageSanitizer
from vk_bot.state.message_dedup import create_message_dedup_store
from vk_bot.state.upload_naming import create_upload_naming_store
from vk_bot.state.user_queue import BatchPoller, UserUploadQueueStore
from vk_bot.state.user_session import create_user_session_store
from vk_bot.logging_setup import setup_logging
from vk_bot.transport.factory import create_vk_transport
from vk_bot.vk.api import VkApiClient
from vk_bot.vk.message_enricher import MessageEnricher

logger = get_logger(__name__)


async def _run() -> None:
  settings = get_settings()
  setup_logging(settings)
  logger.info("Starting VK bot", extra={"event": "bot.start"})
  redis_client = aioredis.from_url(settings.effective_redis_url, decode_responses=True)

  vk_api = VkApiClient(settings)
  core_api = CoreApiClient(settings)
  rate_limiter = create_rate_limiter(settings, redis_client)
  sanitizer = MessageSanitizer(settings)
  queue_store = UserUploadQueueStore(settings, redis_client)
  naming_store = create_upload_naming_store(settings, redis_client)
  message_enricher = MessageEnricher(vk_api, settings)

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
