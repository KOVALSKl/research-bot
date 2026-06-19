from __future__ import annotations

import asyncio
from typing import Any, Callable

import httpx

from research_shared.logging_config import get_logger

logger = get_logger(__name__)


class VkDocUploader:
  """Upload a document to VK via docs.getUploadServer → POST → docs.save."""

  def __init__(
    self,
    get_upload_server: Callable[..., dict[str, Any]],
    save_doc: Callable[..., list[dict[str, Any]]],
  ) -> None:
    self._get_upload_server = get_upload_server
    self._save_doc = save_doc

  async def upload(
    self,
    content: bytes,
    filename: str,
    *,
    peer_id: int,
    timeout_seconds: float = 120.0,
  ) -> str:
    upload_info = await asyncio.to_thread(
      self._get_upload_server,
      type="doc",
      peer_id=peer_id,
    )
    upload_url = upload_info.get("upload_url")
    if not upload_url:
      raise RuntimeError(f"docs.getUploadServer returned no upload_url: {upload_info}")

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
      response = await client.post(
        upload_url,
        files={"file": (filename, content, "application/pdf")},
      )
      response.raise_for_status()
      upload_result = response.json()

    saved = await asyncio.to_thread(
      self._save_doc,
      file=upload_result.get("file", ""),
      title=filename,
    )
    if not saved:
      raise RuntimeError("docs.save returned empty response")

    doc = saved[0]
    owner_id = doc.get("owner_id")
    doc_id = doc.get("id")
    if owner_id is None or doc_id is None:
      raise RuntimeError(f"docs.save returned invalid doc payload: {doc}")

    return f"doc{owner_id}_{doc_id}"
