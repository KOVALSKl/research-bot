import pytest

from vk_bot.config import VkBotSettings
from vk_bot.domain import IncomingMessage
from vk_bot.vk.message_enricher import MessageEnricher


class MockVkApi:
  def __init__(self) -> None:
    self.get_message_data_calls: list[dict] = []
    self.resolve_calls: list[tuple[int, int]] = []
    self.message_data: dict | None = None
    self.resolved_url = "https://example.com/resolved.pdf"

  async def get_message_data(
    self,
    *,
    message_id: int = 0,
    peer_id: int = 0,
    conversation_message_id: int = 0,
  ) -> dict | None:
    self.get_message_data_calls.append(
      {
        "message_id": message_id,
        "peer_id": peer_id,
        "conversation_message_id": conversation_message_id,
      }
    )
    return self.message_data

  async def resolve_doc_url(self, owner_id: int, doc_id: int) -> str | None:
    self.resolve_calls.append((owner_id, doc_id))
    return self.resolved_url


@pytest.mark.asyncio
async def test_enricher_fetches_message_when_attachments_missing():
  settings = VkBotSettings(vk_message_enrich_enabled=True, vk_docs_resolve_url=True)
  vk = MockVkApi()
  vk.message_data = {
    "text": "with pdf",
    "attachments": [
      {
        "type": "doc",
        "doc": {
          "owner_id": 1,
          "id": 2,
          "title": "paper",
          "ext": "pdf",
          "url": "https://example.com/paper.pdf",
        },
      }
    ],
  }
  enricher = MessageEnricher(vk, settings)
  message = IncomingMessage(user_id=1, peer_id=1, message_id=99)

  enriched = await enricher.enrich(message, None)

  assert vk.get_message_data_calls
  assert enriched.text == "with pdf"
  assert len(enriched.attachments) == 1
  assert enriched.attachments[0].ext == "pdf"


@pytest.mark.asyncio
async def test_enricher_resolves_doc_url_when_missing():
  settings = VkBotSettings(vk_message_enrich_enabled=False, vk_docs_resolve_url=True)
  vk = MockVkApi()
  enricher = MessageEnricher(vk, settings)
  raw = [
    {
      "type": "doc",
      "doc": {"owner_id": 5, "id": 6, "title": "paper", "ext": "pdf"},
    }
  ]
  message = IncomingMessage(user_id=1, peer_id=1)

  enriched = await enricher.enrich(message, raw)

  assert vk.resolve_calls == [(5, 6)]
  assert enriched.attachments[0].url == "https://example.com/resolved.pdf"
