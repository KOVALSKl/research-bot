import json
import pytest
from httpx import ConnectError, MockTransport, Request, Response

from vk_bot.config import VkBotSettings
from vk_bot.core_api.client import CoreApiClient, CoreApiError


@pytest.fixture
def settings() -> VkBotSettings:
  return VkBotSettings(
    core_api_base_url="http://testserver",
    vk_core_api_retry_max=3,
    vk_core_api_retry_backoff_seconds=0.01,
  )


def _client_with_transport(settings: VkBotSettings, handler) -> CoreApiClient:
  transport = MockTransport(handler)
  import httpx

  http_client = httpx.AsyncClient(
    base_url=settings.core_api_base_url,
    transport=transport,
  )
  return CoreApiClient(settings, client=http_client)


@pytest.mark.asyncio
async def test_agent_ask_success(settings):
  captured: dict = {}

  def handler(request: Request) -> Response:
    assert request.url.path == "/agent/ask"
    captured["body"] = json.loads(request.content.decode())
    return Response(
      200,
      json={
        "mode": "question",
        "answer": "ok",
        "idea_assessment": None,
        "sources": {"local": [], "external": []},
        "steps": [],
      },
    )

  client = _client_with_transport(settings, handler)
  response = await client.agent_ask("What is AI?")
  assert response.answer == "ok"
  assert captured["body"]["limit"] == 10
  assert captured["body"]["mode"] == "question"
  await client.aclose()


@pytest.mark.asyncio
async def test_ask_success(settings):
  captured: dict = {}

  def handler(request: Request) -> Response:
    assert request.url.path == "/ask"
    captured["body"] = json.loads(request.content.decode())
    return Response(200, json={"answer": "ok", "citations": [], "context_chunks": []})

  client = _client_with_transport(settings, handler)
  response = await client.ask("What is AI?")
  assert response.answer == "ok"
  assert captured["body"]["limit"] == 10
  await client.aclose()


@pytest.mark.asyncio
async def test_ask_http_error(settings):
  def handler(request: Request) -> Response:
    return Response(500, text="internal error")

  client = _client_with_transport(settings, handler)
  with pytest.raises(CoreApiError) as exc:
    await client.ask("question long enough")
  assert exc.value.status_code == 500
  await client.aclose()


@pytest.mark.asyncio
async def test_ask_retries_connect_error_then_succeeds(settings):
  attempts = {"count": 0}

  def handler(request: Request) -> Response:
    attempts["count"] += 1
    if attempts["count"] == 1:
      raise ConnectError("connection refused")
    return Response(200, json={"answer": "ok", "citations": [], "context_chunks": []})

  client = _client_with_transport(settings, handler)
  response = await client.ask("question long enough")
  assert response.answer == "ok"
  assert attempts["count"] == 2
  await client.aclose()


@pytest.mark.asyncio
async def test_ask_retries_exhausted(settings):
  def handler(_request: Request) -> Response:
    raise ConnectError("connection refused")

  client = _client_with_transport(settings, handler)
  with pytest.raises(CoreApiError):
    await client.ask("question long enough")
  await client.aclose()


@pytest.mark.asyncio
async def test_upload_document(settings):
  def handler(request: Request) -> Response:
    assert request.url.path == "/documents"
    return Response(202, json={"task_id": "t1", "research_id": "r1"})

  client = _client_with_transport(settings, handler)
  result = await client.upload_document(b"%PDF", "doc.pdf")
  assert result["task_id"] == "t1"
  await client.aclose()


@pytest.mark.asyncio
async def test_upload_document_retries_read_timeout_then_succeeds(settings):
  import httpx

  attempts = {"count": 0}

  def handler(request: Request) -> Response:
    attempts["count"] += 1
    if attempts["count"] == 1:
      raise httpx.ReadTimeout("read timeout")
    return Response(202, json={"task_id": "t1", "research_id": "r1"})

  client = _client_with_transport(settings, handler)
  result = await client.upload_document(b"%PDF", "doc.pdf")
  assert result["task_id"] == "t1"
  assert attempts["count"] == 2
  await client.aclose()


@pytest.mark.asyncio
async def test_upload_batch(settings):
  def handler(request: Request) -> Response:
    assert request.url.path == "/documents/batch"
    return Response(
      202,
      json={
        "jobs": [
          {"task_id": "t1", "filename": "a.pdf"},
          {"task_id": "t2", "filename": "b.pdf"},
        ]
      },
    )

  client = _client_with_transport(settings, handler)
  result = await client.upload_batch([(b"a", "a.pdf"), (b"b", "b.pdf")])
  assert len(result["jobs"]) == 2
  await client.aclose()


@pytest.mark.asyncio
async def test_upload_batch_error(settings):
  def handler(request: Request) -> Response:
    return Response(422, text="validation error")

  client = _client_with_transport(settings, handler)
  with pytest.raises(CoreApiError) as exc:
    await client.upload_batch([(b"a", "a.pdf")])
  assert exc.value.status_code == 422
  await client.aclose()


@pytest.mark.asyncio
async def test_get_task_status(settings):
  def handler(request: Request) -> Response:
    assert request.url.path == "/documents/tasks/task-1"
    return Response(200, json={"status": "indexed"})

  client = _client_with_transport(settings, handler)
  result = await client.get_task_status("task-1")
  assert result["status"] == "indexed"
  await client.aclose()


@pytest.mark.asyncio
async def test_download_source_file(settings):
  def handler(request: Request) -> Response:
    assert request.url.path == "/documents/files/r1"
    return Response(
      200,
      content=b"%PDF",
      headers={"content-disposition": 'attachment; filename="paper.pdf"'},
    )

  client = _client_with_transport(settings, handler)
  content, filename = await client.download_source_file("r1")
  assert content == b"%PDF"
  assert filename == "paper.pdf"
  await client.aclose()


@pytest.mark.asyncio
async def test_download_source_file_not_found(settings):
  def handler(request: Request) -> Response:
    return Response(404, text="not found")

  client = _client_with_transport(settings, handler)
  with pytest.raises(CoreApiError) as exc:
    await client.download_source_file("missing")
  assert exc.value.status_code == 404
  await client.aclose()


@pytest.mark.asyncio
async def test_list_documents(settings):
  def handler(request: Request) -> Response:
    assert request.url.path == "/documents"
    return Response(
      200,
      json={
        "documents": [
          {
            "research_id": "r1",
            "filename": "paper.pdf",
            "display_name": "My Paper",
            "status": "indexed",
            "chunk_count": 5,
            "indexed_at": None,
          }
        ]
      },
    )

  client = _client_with_transport(settings, handler)
  items = await client.list_documents()
  assert items[0].display_name == "My Paper"
  await client.aclose()


@pytest.mark.asyncio
async def test_agent_stream_dispatches_reasoning_events(settings):
  sse_body = (
    "event: reasoning\n"
    f"data: {json.dumps({'iteration': 1, 'max_iterations': 6, 'thought': 'Thinking', 'action': 'finish'})}\n\n"
    "event: complete\n"
    f"data: {json.dumps({'mode': 'question', 'answer': 'done', 'idea_assessment': None, 'sources': {'local': [], 'external': []}, 'steps': []})}\n\n"
  )

  def handler(request: Request) -> Response:
    assert request.url.path == "/agent/ask/stream"
    return Response(200, content=sse_body, headers={"content-type": "text/event-stream"})

  client = _client_with_transport(settings, handler)
  reasoning_events = []

  async def on_reasoning(event):
    reasoning_events.append(event)

  response = await client.agent_ask_stream(
    "What is AI?",
    on_reasoning=on_reasoning,
  )
  assert response.answer == "done"
  assert len(reasoning_events) == 1
  assert reasoning_events[0].thought == "Thinking"
  await client.aclose()


@pytest.mark.asyncio
async def test_agent_stream_retries_before_first_sse_byte(settings):
  attempts = {"count": 0}
  sse_body = (
    "event: complete\n"
    f"data: {json.dumps({'mode': 'question', 'answer': 'done', 'idea_assessment': None, 'sources': {'local': [], 'external': []}, 'steps': []})}\n\n"
  )

  def handler(request: Request) -> Response:
    attempts["count"] += 1
    if attempts["count"] == 1:
      raise ConnectError("connection refused")
    return Response(200, content=sse_body, headers={"content-type": "text/event-stream"})

  client = _client_with_transport(settings, handler)
  response = await client.agent_ask_stream("What is AI?")
  assert response.answer == "done"
  assert attempts["count"] == 2
  await client.aclose()


@pytest.mark.asyncio
async def test_agent_stream_no_retry_after_first_sse_byte(settings):
  from contextlib import asynccontextmanager
  from unittest.mock import MagicMock

  stream_calls = {"count": 0}

  mock_response = MagicMock()

  async def aiter_lines():
    yield "event: progress"
    raise ConnectError("connection refused after first line")

  mock_response.aiter_lines = aiter_lines
  mock_response.raise_for_status = MagicMock()

  @asynccontextmanager
  async def mock_stream(*_args, **_kwargs):
    stream_calls["count"] += 1
    yield mock_response

  mock_client = MagicMock()
  mock_client.stream = mock_stream

  client = CoreApiClient(settings, client=mock_client)
  with pytest.raises(CoreApiError):
    await client.agent_ask_stream("What is AI?")

  assert stream_calls["count"] == 1


@pytest.mark.asyncio
async def test_upload_document_with_display_name(settings):
  captured: dict = {}

  def handler(request: Request) -> Response:
    captured["content_type"] = request.headers.get("content-type", "")
    return Response(202, json={"task_id": "t1", "research_id": "r1"})

  client = _client_with_transport(settings, handler)
  await client.upload_document(b"%PDF", "doc.pdf", display_name="Custom Name")
  assert "multipart/form-data" in captured["content_type"]
  await client.aclose()
