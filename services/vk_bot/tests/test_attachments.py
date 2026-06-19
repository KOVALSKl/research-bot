import pytest

from vk_bot.vk.attachments import parse_doc_attachments


@pytest.mark.asyncio
async def test_parse_doc_attachments_pdf_with_url():
  raw = [
    {
      "type": "doc",
      "doc": {
        "owner_id": 1,
        "id": 2,
        "title": "paper",
        "ext": "pdf",
        "url": "https://example.com/paper.pdf",
        "size": 1024,
      },
    }
  ]
  attachments = await parse_doc_attachments(raw)
  assert len(attachments) == 1
  assert attachments[0].ext == "pdf"
  assert attachments[0].filename == "paper.pdf"
  assert attachments[0].owner_id == 1
  assert attachments[0].doc_id == 2


@pytest.mark.asyncio
async def test_parse_doc_attachments_skips_non_pdf():
  raw = [
    {
      "type": "doc",
      "doc": {"title": "photo", "ext": "jpg", "url": "https://example.com/x.jpg"},
    }
  ]
  attachments = await parse_doc_attachments(raw)
  assert attachments == []


@pytest.mark.asyncio
async def test_parse_doc_attachments_resolves_missing_url():
  async def resolver(owner_id: int, doc_id: int) -> str | None:
    assert owner_id == 10
    assert doc_id == 20
    return "https://example.com/resolved.pdf"

  raw = [
    {
      "type": "doc",
      "doc": {"owner_id": 10, "id": 20, "title": "paper", "ext": "pdf", "size": 100},
    }
  ]
  attachments = await parse_doc_attachments(raw, doc_url_resolver=resolver)
  assert len(attachments) == 1
  assert attachments[0].url == "https://example.com/resolved.pdf"


@pytest.mark.asyncio
async def test_parse_doc_attachments_keeps_pdf_without_url_when_ids_present():
  raw = [
    {
      "type": "doc",
      "doc": {"owner_id": 10, "id": 20, "title": "paper", "ext": "pdf", "size": 100},
    }
  ]
  attachments = await parse_doc_attachments(raw, resolve_urls=False)
  assert len(attachments) == 1
  assert attachments[0].url == ""
  assert attachments[0].owner_id == 10


@pytest.mark.asyncio
async def test_parse_doc_attachments_normalizes_dict():
  raw = {
    "type": "doc",
    "doc": {"owner_id": 1, "id": 2, "title": "paper", "ext": "pdf", "url": "http://x"},
  }
  attachments = await parse_doc_attachments(raw)
  assert len(attachments) == 1
