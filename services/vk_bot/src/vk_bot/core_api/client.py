from __future__ import annotations

from typing import Any

import httpx
from research_shared.domain.models import AskQuery, AskResponse, DocumentListItem

from vk_bot.config import VkBotSettings


class CoreApiError(Exception):
  def __init__(self, message: str, status_code: int | None = None) -> None:
    super().__init__(message)
    self.status_code = status_code


class CoreApiClient:
  def __init__(
    self,
    settings: VkBotSettings,
    client: httpx.AsyncClient | None = None,
  ) -> None:
    self._settings = settings
    self._owns_client = client is None
    self._client = client or httpx.AsyncClient(
      base_url=settings.core_api_base_url.rstrip("/"),
      timeout=settings.vk_core_api_timeout_seconds,
      headers={"X-Client": "vk_bot"},
    )

  async def aclose(self) -> None:
    if self._owns_client:
      await self._client.aclose()

  async def ask(self, question: str, limit: int | None = None) -> AskResponse:
    payload = AskQuery(
      question=question,
      limit=limit or self._settings.vk_ask_default_limit,
    )
    try:
      response = await self._client.post(
        "/ask",
        json=payload.model_dump(),
        timeout=self._settings.vk_ask_timeout_seconds,
      )
      response.raise_for_status()
    except httpx.HTTPStatusError as exc:
      raise CoreApiError(
        f"Ask request failed: {exc.response.text}",
        status_code=exc.response.status_code,
      ) from exc
    except httpx.HTTPError as exc:
      raise CoreApiError(f"Ask request failed: {exc}") from exc
    return AskResponse.model_validate(response.json())

  async def upload_document(
    self,
    content: bytes,
    filename: str,
    *,
    display_name: str | None = None,
  ) -> dict[str, Any]:
    files = {"file": (filename, content, "application/pdf")}
    data: dict[str, str] = {}
    if display_name:
      data["display_name"] = display_name
    try:
      response = await self._client.post("/documents", files=files, data=data or None)
      response.raise_for_status()
    except httpx.HTTPStatusError as exc:
      raise CoreApiError(
        f"Upload failed: {exc.response.text}",
        status_code=exc.response.status_code,
      ) from exc
    except httpx.HTTPError as exc:
      raise CoreApiError(f"Upload failed: {exc}") from exc
    return response.json()

  async def upload_batch(
    self,
    files: list[tuple[bytes, str]],
    *,
    display_names: list[str] | None = None,
  ) -> dict[str, Any]:
    multipart: list[tuple[str, tuple[str, bytes, str] | str]] = [
      ("files", (filename, content, "application/pdf"))
      for content, filename in files
    ]
    if display_names:
      for name in display_names:
        multipart.append(("display_names", name))
    try:
      response = await self._client.post("/documents/batch", files=multipart)
      response.raise_for_status()
    except httpx.HTTPStatusError as exc:
      raise CoreApiError(
        f"Batch upload failed: {exc.response.text}",
        status_code=exc.response.status_code,
      ) from exc
    except httpx.HTTPError as exc:
      raise CoreApiError(f"Batch upload failed: {exc}") from exc
    return response.json()

  async def list_documents(
    self,
    status: str | None = None,
  ) -> list[DocumentListItem]:
    params: dict[str, str] = {}
    if status:
      params["status"] = status
    try:
      response = await self._client.get("/documents", params=params or None)
      response.raise_for_status()
    except httpx.HTTPStatusError as exc:
      raise CoreApiError(
        f"List documents failed: {exc.response.text}",
        status_code=exc.response.status_code,
      ) from exc
    except httpx.HTTPError as exc:
      raise CoreApiError(f"List documents failed: {exc}") from exc
    payload = response.json()
    return [DocumentListItem.model_validate(item) for item in payload.get("documents", [])]

  async def get_task_status(self, task_id: str) -> dict[str, Any]:
    try:
      response = await self._client.get(f"/documents/tasks/{task_id}")
      response.raise_for_status()
    except httpx.HTTPStatusError as exc:
      raise CoreApiError(
        f"Task status failed: {exc.response.text}",
        status_code=exc.response.status_code,
      ) from exc
    except httpx.HTTPError as exc:
      raise CoreApiError(f"Task status failed: {exc}") from exc
    data = response.json()
    status = data.get("status") or data.get("state", "").lower()
    data["status"] = status
    return data

  async def download_source_file(self, research_id: str) -> tuple[bytes, str]:
    try:
      response = await self._client.get(f"/documents/files/{research_id}")
      response.raise_for_status()
    except httpx.HTTPStatusError as exc:
      raise CoreApiError(
        f"Source file download failed: {exc.response.text}",
        status_code=exc.response.status_code,
      ) from exc
    except httpx.HTTPError as exc:
      raise CoreApiError(f"Source file download failed: {exc}") from exc

    filename = "document.pdf"
    content_disposition = response.headers.get("content-disposition", "")
    if "filename=" in content_disposition:
      filename = content_disposition.split("filename=", 1)[1].strip('"')
    return response.content, filename
