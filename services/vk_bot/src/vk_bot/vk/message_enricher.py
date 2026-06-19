from __future__ import annotations

import logging
from dataclasses import replace

from vk_bot.config import VkBotSettings
from vk_bot.domain import IncomingMessage
from vk_bot.vk.api import VkApiClient
from vk_bot.vk.attachments import parse_doc_attachments

logger = logging.getLogger(__name__)


class MessageEnricher:
  def __init__(self, vk_client: VkApiClient, settings: VkBotSettings) -> None:
    self._vk = vk_client
    self._settings = settings

  async def enrich(
    self,
    message: IncomingMessage,
    raw_attachments: object,
  ) -> IncomingMessage:
    attachments_raw = raw_attachments
    text = message.text

    if self._settings.vk_message_enrich_enabled and self._needs_message_enrichment(attachments_raw):
      try:
        data = await self._vk.get_message_data(
          message_id=message.message_id,
          peer_id=message.peer_id,
          conversation_message_id=message.conversation_message_id,
        )
      except Exception:
        logger.exception(
          "Failed to enrich message peer_id=%s message_id=%s",
          message.peer_id,
          message.message_id,
        )
        data = None

      if isinstance(data, dict):
        attachments_raw = data.get("attachments", attachments_raw)
        if not text:
          text = str(data.get("text", "") or "")

    resolver = self._vk.resolve_doc_url if self._settings.vk_docs_resolve_url else None
    attachments = await parse_doc_attachments(
      attachments_raw,
      doc_url_resolver=resolver,
      resolve_urls=self._settings.vk_docs_resolve_url,
    )
    return replace(message, text=text, attachments=attachments)

  def _needs_message_enrichment(self, raw_attachments: object) -> bool:
    if raw_attachments is None:
      return True
    if isinstance(raw_attachments, dict):
      return True
    if isinstance(raw_attachments, list):
      return len(raw_attachments) == 0
    return True
