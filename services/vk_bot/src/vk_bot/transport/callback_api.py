from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response

from vk_bot.config import VkBotSettings
from vk_bot.domain import IncomingMessage
from vk_bot.vk.message_enricher import MessageEnricher
from vk_bot.vk.outgoing_filter import is_outgoing

logger = logging.getLogger(__name__)


def _callback_to_message(payload: dict) -> tuple[IncomingMessage | None, object]:
  obj = payload.get("object")
  if not isinstance(obj, dict):
    return None, None
  message = obj.get("message", obj)
  if not isinstance(message, dict):
    return None, None

  from_id = int(message.get("from_id") or message.get("user_id") or 0)
  out = int(message.get("out") or 0)
  if is_outgoing(from_me=False, out=out, from_id=from_id):
    return None, None

  peer_id = int(message.get("peer_id") or from_id)
  text = str(message.get("text", "") or "")
  message_id = int(message.get("id") or 0)
  conversation_message_id = int(message.get("conversation_message_id") or 0)
  raw_attachments = message.get("attachments")

  return (
    IncomingMessage(
      user_id=from_id,
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


class VkCallbackTransport:
  def __init__(
    self,
    settings: VkBotSettings,
    message_enricher: MessageEnricher | None = None,
  ) -> None:
    self._settings = settings
    self._message_enricher = message_enricher
    self._app = FastAPI()
    self._handler: Callable[[IncomingMessage], Awaitable[None]] | None = None
    self._server: uvicorn.Server | None = None
    self._register_routes()

  def _register_routes(self) -> None:
    @self._app.post(self._settings.vk_callback_path)
    async def callback_endpoint(request: Request) -> Response:
      payload = await request.json()
      event_type = payload.get("type")

      if event_type == "confirmation":
        return Response(
          content=self._settings.vk_callback_confirmation,
          media_type="text/plain",
        )

      secret = payload.get("secret", "")
      if self._settings.vk_callback_secret and secret != self._settings.vk_callback_secret:
        raise HTTPException(status_code=403, detail="Invalid secret")

      if event_type == "message_new" and self._handler is not None:
        message, raw_attachments = _callback_to_message(payload)
        if message is not None:
          asyncio.create_task(self._process_message(message, raw_attachments))

      return Response(content="ok", media_type="text/plain")

  async def _process_message(
    self,
    message: IncomingMessage,
    raw_attachments: object,
  ) -> None:
    if self._handler is None:
      return
    if self._message_enricher is not None:
      message = await self._message_enricher.enrich(message, raw_attachments)
    await self._handler(message)

  async def run(
    self,
    handler: Callable[[IncomingMessage], Awaitable[None]],
  ) -> None:
    self._handler = handler
    config = uvicorn.Config(
      self._app,
      host=self._settings.vk_callback_host,
      port=self._settings.vk_callback_port,
      log_level="info",
    )
    self._server = uvicorn.Server(config)
    await self._server.serve()

  async def stop(self) -> None:
    if self._server is not None:
      self._server.should_exit = True
