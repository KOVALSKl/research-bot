import pytest
import httpx

from vk_bot.config import VkBotSettings
from vk_bot.domain import Attachment
from vk_bot.vk.api import VkApiClient


@pytest.mark.asyncio
async def test_download_follows_redirect():
  settings = VkBotSettings(vk_bot_token="token")
  client = VkApiClient(settings)
  attachment = Attachment(
    filename="paper.pdf",
    url="https://vk.com/doc1_2",
    ext="pdf",
  )

  async def handler(request: httpx.Request) -> httpx.Response:
    if str(request.url).startswith("https://vk.com/"):
      return httpx.Response(
        302,
        headers={"Location": "https://cdn.example.com/paper.pdf"},
      )
    return httpx.Response(200, content=b"%PDF-1.4 content")

  original_async_client = httpx.AsyncClient

  class TrackingClient(original_async_client):
    def __init__(self, *args, **kwargs):
      kwargs["transport"] = httpx.MockTransport(handler)
      kwargs["follow_redirects"] = True
      super().__init__(*args, **kwargs)

  httpx.AsyncClient = TrackingClient  # type: ignore[misc,assignment]
  try:
    files = await client.download_attachments([attachment])
  finally:
    httpx.AsyncClient = original_async_client  # type: ignore[misc]

  assert files == [(b"%PDF-1.4 content", "paper.pdf")]


@pytest.mark.asyncio
async def test_download_redirect_chain_userapi():
  settings = VkBotSettings(vk_bot_token="token")
  client = VkApiClient(settings)
  attachment = Attachment(
    filename="paper.pdf",
    url="https://vk.com/doc1_2",
    ext="pdf",
  )
  requests_hosts: list[str] = []

  async def handler(request: httpx.Request) -> httpx.Response:
    host = httpx.URL(str(request.url)).host or ""
    requests_hosts.append(host)
    if host == "vk.com":
      return httpx.Response(
        302,
        headers={
          "Location": "https://psv4.userapi.com/file.pdf",
          "Set-Cookie": "remixlang=0; Path=/",
        },
      )
    return httpx.Response(200, content=b"%PDF userapi chain")

  original_async_client = httpx.AsyncClient

  class TrackingClient(original_async_client):
    def __init__(self, *args, **kwargs):
      kwargs["transport"] = httpx.MockTransport(handler)
      kwargs["follow_redirects"] = True
      super().__init__(*args, **kwargs)

  httpx.AsyncClient = TrackingClient  # type: ignore[misc,assignment]
  try:
    files = await client.download_attachments([attachment])
  finally:
    httpx.AsyncClient = original_async_client  # type: ignore[misc]

  assert files == [(b"%PDF userapi chain", "paper.pdf")]
  assert requests_hosts[0] == "vk.com"
  assert "psv4.userapi.com" in requests_hosts[-1]


@pytest.mark.asyncio
async def test_download_302_without_location_raises():
  settings = VkBotSettings(vk_bot_token="token")
  client = VkApiClient(settings)
  attachment = Attachment(
    filename="paper.pdf",
    url="https://vk.com/doc1_2",
    ext="pdf",
  )

  async def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(302, headers={})

  original_async_client = httpx.AsyncClient

  class TrackingClient(original_async_client):
    def __init__(self, *args, **kwargs):
      kwargs["transport"] = httpx.MockTransport(handler)
      kwargs["follow_redirects"] = True
      super().__init__(*args, **kwargs)

  httpx.AsyncClient = TrackingClient  # type: ignore[misc,assignment]
  try:
    with pytest.raises(httpx.HTTPStatusError):
      await client.download_attachments([attachment])
  finally:
    httpx.AsyncClient = original_async_client  # type: ignore[misc]


@pytest.mark.asyncio
async def test_download_multiple_attachments_sequentially():
  settings = VkBotSettings(vk_bot_token="token")
  client = VkApiClient(settings)
  calls: list[str] = []

  async def handler(request: httpx.Request) -> httpx.Response:
    calls.append(str(request.url))
    name = str(request.url).split("/")[-1]
    return httpx.Response(200, content=f"%PDF {name}".encode())

  original_async_client = httpx.AsyncClient

  class TrackingClient(original_async_client):
    def __init__(self, *args, **kwargs):
      kwargs["transport"] = httpx.MockTransport(handler)
      kwargs["follow_redirects"] = True
      super().__init__(*args, **kwargs)

  attachments = [
    Attachment(filename="a.pdf", url="https://cdn.example.com/a.pdf", ext="pdf"),
    Attachment(filename="b.pdf", url="https://cdn.example.com/b.pdf", ext="pdf"),
  ]
  httpx.AsyncClient = TrackingClient  # type: ignore[misc,assignment]
  try:
    files = await client.download_attachments(attachments)
  finally:
    httpx.AsyncClient = original_async_client  # type: ignore[misc]

  assert len(files) == 2
  assert calls == [
    "https://cdn.example.com/a.pdf",
    "https://cdn.example.com/b.pdf",
  ]
