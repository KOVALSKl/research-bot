from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from research_shared.agents.models import AgentAskRequest, AgentAskResponse, AgentProgressEvent, AgentReasoningEvent
from research_shared.domain.models import AskQuery, AskResponse, DocumentListItem

from vk_bot.config import VkBotSettings

_RETRYABLE_ERRORS = (
  httpx.ConnectError,
  httpx.ConnectTimeout,
  httpx.ReadTimeout,
  httpx.RemoteProtocolError,
)


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

  async def _request_with_retry(
    self,
    method: str,
    url: str,
    *,
    error_prefix: str,
    **kwargs: Any,
  ) -> httpx.Response:
    max_attempts = self._settings.vk_core_api_retry_max
    base_delay = self._settings.vk_core_api_retry_backoff_seconds
    last_exc: Exception | None = None

    for attempt in range(max_attempts):
      try:
        response = await self._client.request(method, url, **kwargs)
        response.raise_for_status()
        return response
      except httpx.HTTPStatusError as exc:
        raise CoreApiError(
          f"{error_prefix}: {exc.response.text}",
          status_code=exc.response.status_code,
        ) from exc
      except _RETRYABLE_ERRORS as exc:
        last_exc = exc
        if attempt + 1 >= max_attempts:
          break
        await asyncio.sleep(base_delay * (2**attempt))
      except httpx.HTTPError as exc:
        raise CoreApiError(f"{error_prefix}: {exc}") from exc

    assert last_exc is not None
    raise CoreApiError(f"{error_prefix}: {last_exc}") from last_exc

  async def ask(self, question: str, limit: int | None = None) -> AskResponse:
    payload = AskQuery(
      question=question,
      limit=limit or self._settings.vk_ask_default_limit,
    )
    response = await self._request_with_retry(
      "POST",
      "/ask",
      error_prefix="Ask request failed",
      json=payload.model_dump(),
      timeout=self._settings.vk_ask_timeout_seconds,
    )
    return AskResponse.model_validate(response.json())

  async def agent_ask(
    self,
    message: str,
    mode: str = "question",
    limit: int | None = None,
    conversation_history: list[dict[str, str]] | None = None,
  ) -> AgentAskResponse:
    payload = AgentAskRequest(
      message=message,
      mode=mode,
      limit=limit or self._settings.vk_ask_default_limit,
      conversation_history=conversation_history or [],
    )
    response = await self._request_with_retry(
      "POST",
      "/agent/ask",
      error_prefix="Agent ask request failed",
      json=payload.model_dump(),
      timeout=self._settings.vk_ask_timeout_seconds,
    )
    return AgentAskResponse.model_validate(response.json())

  async def agent_ask_stream(
    self,
    message: str,
    mode: str = "question",
    limit: int | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    *,
    on_progress: Callable[[AgentProgressEvent], Awaitable[None] | None] | None = None,
    on_reasoning: Callable[[AgentReasoningEvent], Awaitable[None] | None] | None = None,
  ) -> AgentAskResponse:
    payload = AgentAskRequest(
      message=message,
      mode=mode,
      limit=limit or self._settings.vk_ask_default_limit,
      conversation_history=conversation_history or [],
    )
    timeout = httpx.Timeout(
      connect=self._settings.vk_core_api_timeout_seconds,
      read=self._settings.vk_ask_timeout_seconds,
      write=self._settings.vk_core_api_timeout_seconds,
      pool=self._settings.vk_core_api_timeout_seconds,
    )
    max_attempts = self._settings.vk_core_api_retry_max
    base_delay = self._settings.vk_core_api_retry_backoff_seconds
    last_exc: Exception | None = None

    for attempt in range(max_attempts):
      stream_started = False
      try:
        # Retry only before the first SSE byte; mid-stream errors must not restart.
        async with self._client.stream(
          "POST",
          "/agent/ask/stream",
          json=payload.model_dump(),
          timeout=timeout,
        ) as response:
          response.raise_for_status()
          final_response_holder: dict[str, AgentAskResponse | None] = {"response": None}
          event_name: str | None = None
          data_lines: list[str] = []

          async for line in response.aiter_lines():
            stream_started = True
            if line.startswith("event:"):
              if event_name and data_lines:
                await self._dispatch_sse_event(
                  event_name,
                  "\n".join(data_lines),
                  on_progress,
                  on_reasoning,
                  final_response_holder,
                )
              event_name = line.split(":", 1)[1].strip()
              data_lines = []
              continue
            if line.startswith("data:"):
              data_lines.append(line.split(":", 1)[1].strip())
              continue
            if line == "" and event_name and data_lines:
              await self._dispatch_sse_event(
                event_name,
                "\n".join(data_lines),
                on_progress,
                on_reasoning,
                final_response_holder,
              )
              event_name = None
              data_lines = []

          if event_name and data_lines:
            await self._dispatch_sse_event(
              event_name,
              "\n".join(data_lines),
              on_progress,
              on_reasoning,
              final_response_holder,
            )

          final_response = final_response_holder["response"]
          if final_response is None:
            raise CoreApiError("Agent stream ended without complete event")
          return final_response
      except httpx.HTTPStatusError as exc:
        raise CoreApiError(
          f"Agent stream request failed: {exc.response.text}",
          status_code=exc.response.status_code,
        ) from exc
      except CoreApiError:
        raise
      except _RETRYABLE_ERRORS as exc:
        last_exc = exc
        if stream_started or attempt + 1 >= max_attempts:
          break
        await asyncio.sleep(base_delay * (2**attempt))
      except httpx.HTTPError as exc:
        raise CoreApiError(f"Agent stream request failed: {exc}") from exc

    assert last_exc is not None
    raise CoreApiError(f"Agent stream request failed: {last_exc}") from last_exc

  @staticmethod
  async def _dispatch_sse_event(
    event_name: str,
    payload_text: str,
    on_progress: Callable[[AgentProgressEvent], Awaitable[None] | None] | None,
    on_reasoning: Callable[[AgentReasoningEvent], Awaitable[None] | None] | None,
    final_response_holder: dict[str, AgentAskResponse | None],
  ) -> None:
    if event_name == "progress" and on_progress is not None:
      event = AgentProgressEvent.model_validate_json(payload_text)
      result = on_progress(event)
      if asyncio.iscoroutine(result):
        await result
    elif event_name == "reasoning" and on_reasoning is not None:
      event = AgentReasoningEvent.model_validate_json(payload_text)
      result = on_reasoning(event)
      if asyncio.iscoroutine(result):
        await result
    elif event_name == "complete":
      final_response_holder["response"] = AgentAskResponse.model_validate_json(payload_text)
    elif event_name == "error":
      detail = json.loads(payload_text).get("detail", "Agent stream failed")
      raise CoreApiError(str(detail))

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
    response = await self._request_with_retry(
      "POST",
      "/documents",
      error_prefix="Upload failed",
      files=files,
      data=data or None,
    )
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
    response = await self._request_with_retry(
      "POST",
      "/documents/batch",
      error_prefix="Batch upload failed",
      files=multipart,
    )
    return response.json()

  async def list_documents(
    self,
    status: str | None = None,
  ) -> list[DocumentListItem]:
    params: dict[str, str] = {}
    if status:
      params["status"] = status
    response = await self._request_with_retry(
      "GET",
      "/documents",
      error_prefix="List documents failed",
      params=params or None,
    )
    payload = response.json()
    return [DocumentListItem.model_validate(item) for item in payload.get("documents", [])]

  async def get_task_status(self, task_id: str) -> dict[str, Any]:
    response = await self._request_with_retry(
      "GET",
      f"/documents/tasks/{task_id}",
      error_prefix="Task status failed",
    )
    data = response.json()
    status = data.get("status") or data.get("state", "").lower()
    data["status"] = status
    return data

  async def download_source_file(self, research_id: str) -> tuple[bytes, str]:
    response = await self._request_with_retry(
      "GET",
      f"/documents/files/{research_id}",
      error_prefix="Source file download failed",
    )

    filename = "document.pdf"
    content_disposition = response.headers.get("content-disposition", "")
    if "filename=" in content_disposition:
      filename = content_disposition.split("filename=", 1)[1].strip('"')
    return response.content, filename

  async def download_external_pdf(
    self,
    cache_key: str,
    *,
    pdf_url: str | None = None,
  ) -> tuple[bytes, str]:
    params = {"pdf_url": pdf_url} if pdf_url else None
    response = await self._request_with_retry(
      "GET",
      f"/literature/papers/{cache_key}/pdf",
      error_prefix="External PDF download failed",
      params=params,
    )

    filename = "external.pdf"
    content_disposition = response.headers.get("content-disposition", "")
    if "filename=" in content_disposition:
      filename = content_disposition.split("filename=", 1)[1].strip('"')
    return response.content, filename
