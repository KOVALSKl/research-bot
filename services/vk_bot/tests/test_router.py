import pytest

from vk_bot.config import VkBotSettings
from vk_bot.domain import Attachment, IncomingMessage
from vk_bot.handlers.router import CommandRouter, Intent


@pytest.fixture
def router() -> CommandRouter:
  return CommandRouter(VkBotSettings())


def _msg(**kwargs) -> IncomingMessage:
  return IncomingMessage(user_id=1, peer_id=1, **kwargs)


def test_greeting_intents(router: CommandRouter):
  for text in ("привет", "старт", "/start", "начать", "hello", "hi", "Привет"):
    result = router.resolve(_msg(text=text))
    assert result.intent == Intent.GREETING


def test_help_intents(router: CommandRouter):
  for text in ("помощь", "/help", "?", "команды"):
    result = router.resolve(_msg(text=text))
    assert result.intent == Intent.HELP


def test_ask_prefixes(router: CommandRouter):
  cases = [
    ("/ask What is AI?", "What is AI?"),
    ("/вопрос Что такое квантовый компьютер?", "Что такое квантовый компьютер?"),
    ("? Explain transformers", "Explain transformers"),
    ("вопрос: как работает RAG?", "как работает RAG?"),
  ]
  for text, expected in cases:
    result = router.resolve(_msg(text=text))
    assert result.intent == Intent.ASK
    assert result.ask_text == expected


def test_list_intents(router: CommandRouter):
  for text in ("/list", "/исследования", "/research", "/list extra"):
    result = router.resolve(_msg(text=text))
    assert result.intent == Intent.LIST


def test_upload_pdf(router: CommandRouter):
  attachment = Attachment(filename="paper.pdf", url="http://x", ext="pdf")
  result = router.resolve(_msg(text="ignored caption", attachments=[attachment]))
  assert result.intent == Intent.UPLOAD


def test_unknown_text(router: CommandRouter):
  result = router.resolve(_msg(text="What is quantum computing without command?"))
  assert result.intent == Intent.UNKNOWN


def test_ask_without_payload_is_unknown(router: CommandRouter):
  result = router.resolve(_msg(text="/ask"))
  assert result.intent == Intent.UNKNOWN


def test_idea_without_payload_returns_help(router: CommandRouter):
  result = router.resolve(_msg(text="/idea"))
  assert result.intent == Intent.HELP


def test_idea_prefixes(router: CommandRouter):
  cases = [
    ("/idea Use GNN for fraud detection", "Use GNN for fraud detection"),
    ("/идея: применить трансформеры к ценам", "применить трансформеры к ценам"),
  ]
  for text, expected in cases:
    result = router.resolve(_msg(text=text))
    assert result.intent == Intent.IDEA
    assert result.idea_text == expected
