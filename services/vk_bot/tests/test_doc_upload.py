import pytest

from vk_bot.vk.doc_upload import VkDocUploader


@pytest.mark.asyncio
async def test_vk_doc_uploader_returns_attachment_string():
  upload_calls: list[dict] = []

  def get_upload_server(**kwargs):
    upload_calls.append(kwargs)
    return {"upload_url": "https://upload.example.com/"}

  def save_doc(**kwargs):
    return [{"owner_id": 123, "id": 456}]

  uploader = VkDocUploader(get_upload_server, save_doc)

  import httpx

  original_client = httpx.AsyncClient

  class FakeClient:
    def __init__(self, *args, **kwargs):
      pass

    async def __aenter__(self):
      return self

    async def __aexit__(self, *args):
      return None

    async def post(self, url, files):
      class Response:
        def raise_for_status(self):
          return None

        def json(self):
          return {"file": "saved-token"}

      return Response()

  httpx.AsyncClient = FakeClient  # type: ignore[misc,assignment]
  try:
    attachment = await uploader.upload(b"%PDF", "paper.pdf", peer_id=42)
  finally:
    httpx.AsyncClient = original_client  # type: ignore[misc]

  assert attachment == "doc123_456"
  assert upload_calls == [{"type": "doc", "peer_id": 42}]
