from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import vk_api
from vk_api.longpoll import VkEventType, VkLongPoll

from vk_bot.config import VkBotSettings
from vk_bot.domain import IncomingMessage
from vk_bot.vk.message_enricher import MessageEnricher
from vk_bot.vk.outgoing_filter import is_outgoing

logger = logging.getLogger(__name__)


def _extract_message_fields(event: object) -> tuple[dict | None, object]:
  raw_attachments: object = getattr(event, "attachments", None)
  message_data: dict | None = None

  if hasattr(event, "raw"):
    raw = getattr(event, "raw", {})
    if isinstance(raw, dict):
      obj = raw.get("object", {})
      if isinstance(obj, dict):
        message = obj.get("message", obj)
        if isinstance(message, dict):
          message_data = message
          if raw_attachments is None:
            raw_attachments = message.get("attachments")

  return message_data, raw_attachments


def _event_to_message(event: object) -> tuple[IncomingMessage | None, object]:
  if getattr(event, "type", None) != VkEventType.MESSAGE_NEW:
    return None, None

  message_data, raw_attachments = _extract_message_fields(event)

  from_id = int(getattr(event, "from_id", 0) or getattr(event, "user_id", 0) or 0)
  out = int(getattr(event, "out", 0) or 0)
  from_me = bool(getattr(event, "from_me", False))
  user_id = int(getattr(event, "user_id", 0) or getattr(event, "peer_id", 0))
  peer_id = int(getattr(event, "peer_id", 0))
  text = str(getattr(event, "text", "") or "")
  message_id = int(getattr(event, "message_id", 0) or 0)
  conversation_message_id = 0

  if isinstance(message_data, dict):
    from_id = int(message_data.get("from_id") or from_id or 0)
    out = int(message_data.get("out") or out or 0)
    if not text:
      text = str(message_data.get("text", "") or "")
    message_id = int(message_data.get("id") or message_id or 0)
    conversation_message_id = int(message_data.get("conversation_message_id") or 0)

  if is_outgoing(from_me=from_me, out=out, from_id=from_id):
    return None, None

  return (
    IncomingMessage(
      user_id=user_id,
      peer_id=peer_id,
      text=text,
      attachments=[],
      from_id=from_id,
      is_outgoing=False,
      message_id=message_id,
      conversation_message_id=conversation_message_id,
    ),
    raw_attachments,
  )


class VkLongPollingTransport:
  def __init__(
    self,
    settings: VkBotSettings,
    message_enricher: MessageEnricher | None = None,
  ) -> None:
    self._settings = settings
    self._message_enricher = message_enricher
    self._stop_event = asyncio.Event()
    self._thread_stop = asyncio.Event()
    self._loop: asyncio.AbstractEventLoop | None = None
    self._poll_task: asyncio.Task[None] | None = None

  async def run(
    self,
    handler: Callable[[IncomingMessage], Awaitable[None]],
  ) -> None:
    if not self._settings.vk_bot_token:
      raise ValueError("VK_BOT_TOKEN is required for long polling transport")
    if not self._settings.vk_group_id:
      raise ValueError("VK_GROUP_ID is required for long polling transport")

    self._loop = asyncio.get_running_loop()
    self._stop_event.clear()
    self._thread_stop.clear()

    self._poll_task = asyncio.create_task(self._poll_loop(handler))
    try:
      await self._poll_task
    except asyncio.CancelledError:
      pass

  async def stop(self) -> None:
    self._stop_event.set()
    self._thread_stop.set()
    if self._poll_task is not None:
      self._poll_task.cancel()
      try:
        await self._poll_task
      except asyncio.CancelledError:
        pass

  async def _poll_loop(
    self,
    handler: Callable[[IncomingMessage], Awaitable[None]],
  ) -> None:
    while not self._stop_event.is_set():
      try:
        await asyncio.to_thread(self._poll_once, handler)
      except Exception:
        logger.exception("Long poll iteration failed")
        await asyncio.sleep(1)

  def _poll_once(self, handler: Callable[[IncomingMessage], Awaitable[None]]) -> None:
    session = vk_api.VkApi(
      token=self._settings.vk_bot_token,
      api_version=self._settings.vk_api_version,
    )
    longpoll = VkLongPoll(
      session,
      group_id=self._settings.vk_group_id,
      preload_messages=True,
    )

    for event in longpoll.listen():
      if self._thread_stop.is_set():
        break
      message, raw_attachments = _event_to_message(event)
      if message is None:
        continue
      if self._loop is None:
        continue
      future = asyncio.run_coroutine_threadsafe(
        self._process_message(message, raw_attachments, handler),
        self._loop,
      )
      try:
        future.result(timeout=300)
      except Exception:
        logger.exception("Handler failed for long poll message")

  async def _process_message(
    self,
    message: IncomingMessage,
    raw_attachments: object,
    handler: Callable[[IncomingMessage], Awaitable[None]],
  ) -> None:
    if self._message_enricher is not None:
      message = await self._message_enricher.enrich(message, raw_attachments)
    await handler(message)
