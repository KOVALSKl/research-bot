from __future__ import annotations

import asyncio
import random
from typing import Any, Protocol
from urllib.parse import urljoin

import httpx
import vk_api

from research_shared.logging_config import get_logger
from vk_bot.config import VkBotSettings
from vk_bot.domain import Attachment
from vk_bot.vk.doc_upload import VkDocUploader

logger = get_logger(__name__)

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_VK_USER_AGENT = "vk-bot/1.0"


class VkApiClientProtocol(Protocol):
  async def send_message(
    self,
    peer_id: int,
    text: str,
    *,
    attachment: str | None = None,
  ) -> None: ...

  async def download_attachments(
    self,
    attachments: list[Attachment],
  ) -> list[tuple[bytes, str]]: ...

  async def resolve_doc_url(self, owner_id: int, doc_id: int) -> str | None: ...

  async def get_message_data(
    self,
    *,
    message_id: int = 0,
    peer_id: int = 0,
    conversation_message_id: int = 0,
  ) -> dict[str, Any] | None: ...

  async def upload_doc_to_vk(
    self,
    content: bytes,
    filename: str,
    peer_id: int,
  ) -> str: ...


class VkApiClient:
  def __init__(self, settings: VkBotSettings) -> None:
    self._settings = settings
    self._session = vk_api.VkApi(
      token=settings.vk_bot_token,
      api_version=settings.vk_api_version,
    )
    self._api = self._session.get_api()
    self._doc_uploader = VkDocUploader(
      self._api.docs.getUploadServer,
      self._api.docs.save,
    )

  async def send_message(
    self,
    peer_id: int,
    text: str,
    *,
    attachment: str | None = None,
  ) -> None:
    kwargs: dict[str, Any] = {
      "peer_id": peer_id,
      "message": text,
      "random_id": random.randint(1, 2**31 - 1),
    }
    if attachment:
      kwargs["attachment"] = attachment
    await asyncio.to_thread(self._api.messages.send, **kwargs)

  async def get_messages_by_id(self, message_ids: list[int]) -> list[dict[str, Any]]:
    if not message_ids:
      return []

    def _call() -> dict[str, Any]:
      return self._api.messages.getById(
        message_ids=",".join(str(message_id) for message_id in message_ids),
        extended=1,
      )

    result = await asyncio.to_thread(_call)
    items = result.get("items", [])
    return [item for item in items if isinstance(item, dict)]

  async def preload_messages(
    self,
    peer_id: int,
    conversation_message_ids: list[int],
  ) -> list[dict[str, Any]]:
    if not peer_id or not conversation_message_ids:
      return []

    def _call() -> dict[str, Any]:
      return self._api.messages.preload_messages(
        peer_id=peer_id,
        cmids=",".join(str(cmid) for cmid in conversation_message_ids),
      )

    result = await asyncio.to_thread(_call)
    items = result.get("items", [])
    return [item for item in items if isinstance(item, dict)]

  async def get_doc_by_id(self, owner_id: int, doc_id: int) -> dict[str, Any] | None:
    if not owner_id or not doc_id:
      return None

    def _call() -> list[dict[str, Any]]:
      return self._api.docs.getById(docs=f"{owner_id}_{doc_id}")

    try:
      docs = await asyncio.to_thread(_call)
    except Exception:
      logger.exception("docs.getById failed for %s_%s", owner_id, doc_id)
      return None

    if not docs:
      return None
    doc = docs[0]
    return doc if isinstance(doc, dict) else None

  async def resolve_doc_url(self, owner_id: int, doc_id: int) -> str | None:
    doc = await self.get_doc_by_id(owner_id, doc_id)
    if not doc:
      return None
    url = doc.get("url") or doc.get("access_url")
    return str(url) if url else None

  async def get_message_data(
    self,
    *,
    message_id: int = 0,
    peer_id: int = 0,
    conversation_message_id: int = 0,
  ) -> dict[str, Any] | None:
    if message_id:
      items = await self.get_messages_by_id([message_id])
      if items:
        return items[0]

    if peer_id and conversation_message_id:
      items = await self.preload_messages(peer_id, [conversation_message_id])
      if items:
        return items[0]

    return None

  async def upload_doc_to_vk(
    self,
    content: bytes,
    filename: str,
    peer_id: int,
  ) -> str:
    max_bytes = self._settings.vk_ask_attach_max_bytes
    if len(content) > max_bytes:
      logger.warning(
        "Skipping oversized source file for VK doc upload",
        extra={
          "attachment_name": filename,
          "count": len(content),
          "event": "ask.doc_upload_skip_oversized",
        },
      )
      raise ValueError(f"File {filename} exceeds VK_ASK_ATTACH_MAX_BYTES")

    return await self._doc_uploader.upload(
      content,
      filename,
      peer_id=peer_id,
    )

  async def _fetch_with_redirects(
    self,
    client: httpx.AsyncClient,
    url: str,
  ) -> tuple[httpx.Response, int]:
    max_redirects = self._settings.vk_http_max_redirects
    headers = {"User-Agent": _VK_USER_AGENT}
    redirect_count = 0
    current_url = url

    while True:
      response = await client.get(
        current_url,
        follow_redirects=False,
        headers=headers,
      )
      if response.status_code not in _REDIRECT_STATUSES:
        return response, redirect_count

      if redirect_count >= max_redirects:
        raise httpx.TooManyRedirects(
          f"Exceeded max redirects ({max_redirects}) for {url}",
          request=response.request,
        )

      location = response.headers.get("Location")
      if not location:
        return response, redirect_count

      redirect_count += 1
      current_url = urljoin(str(response.url), location)

  async def _download_url(
    self,
    client: httpx.AsyncClient,
    url: str,
    *,
    attachment_name: str,
  ) -> httpx.Response:
    initial_host = httpx.URL(url).host
    redirect_count = 0
    response: httpx.Response | None = None

    try:
      response = await client.get(url)
      redirect_count = len(response.history)
    except httpx.TooManyRedirects:
      response, redirect_count = await self._fetch_with_redirects(client, url)

    if response is not None and response.status_code in _REDIRECT_STATUSES:
      response, manual_count = await self._fetch_with_redirects(client, url)
      redirect_count += manual_count

    if response is None:
      raise httpx.RequestError(f"Failed to download {attachment_name}")

    final_host = httpx.URL(str(response.url)).host
    logger.info(
      "Attachment download response",
      extra={
        "attachment_name": attachment_name,
        "redirect_count": redirect_count,
        "initial_host": initial_host,
        "final_host": final_host,
        "status": response.status_code,
        "event": "attachment.download",
      },
    )

    if response.status_code != 200:
      response.raise_for_status()
    return response

  async def _download_single_attachment(
    self,
    client: httpx.AsyncClient,
    attachment: Attachment,
    *,
    max_bytes: int,
  ) -> tuple[bytes, str] | None:
    if attachment.ext != "pdf":
      return None
    if attachment.size and attachment.size > max_bytes:
      logger.warning(
        "Skipping oversized attachment",
        extra={"attachment_name": attachment.filename, "event": "attachment.skip_oversized"},
      )
      return None

    url = attachment.url
    if not url and self._settings.vk_docs_resolve_url and attachment.owner_id and attachment.doc_id:
      url = await self.resolve_doc_url(attachment.owner_id, attachment.doc_id) or ""

    if not url:
      logger.warning(
        "No download URL for attachment",
        extra={"attachment_name": attachment.filename, "event": "attachment.no_url"},
      )
      return None

    logger.info(
      "Downloading attachment",
      extra={"attachment_name": attachment.filename, "event": "attachment.download_start"},
    )
    response = await self._download_url(
      client,
      url,
      attachment_name=attachment.filename,
    )
    content = response.content
    if len(content) > max_bytes:
      logger.warning(
        "Downloaded attachment exceeds size limit",
        extra={"attachment_name": attachment.filename, "event": "attachment.skip_oversized"},
      )
      return None

    logger.info(
      "Attachment downloaded",
      extra={
        "attachment_name": attachment.filename,
        "count": len(content),
        "final_host": httpx.URL(str(response.url)).host,
        "event": "attachment.download_success",
      },
    )
    return content, attachment.filename

  async def download_attachments(
    self,
    attachments: list[Attachment],
  ) -> list[tuple[bytes, str]]:
    results: list[tuple[bytes, str]] = []
    max_bytes = self._settings.vk_attachment_max_bytes

    async with httpx.AsyncClient(
      timeout=60.0,
      follow_redirects=True,
      max_redirects=self._settings.vk_http_max_redirects,
      headers={"User-Agent": _VK_USER_AGENT},
    ) as client:
      for index, attachment in enumerate(attachments, start=1):
        logger.info(
          "Processing attachment %s/%s",
          index,
          len(attachments),
          extra={
            "attachment_name": attachment.filename,
            "count": index,
            "event": "attachment.download_progress",
          },
        )
        try:
          downloaded = await self._download_single_attachment(
            client,
            attachment,
            max_bytes=max_bytes,
          )
        except Exception:
          logger.exception(
            "Attachment download failed",
            extra={"attachment_name": attachment.filename, "event": "attachment.download_error"},
          )
          raise
        if downloaded is not None:
          results.append(downloaded)
    return results
