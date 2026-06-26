import asyncio

import pytest
from httpx import ASGITransport, AsyncClient
from vk_api.longpoll import VkEventType

from vk_bot.config import VkBotSettings
from vk_bot.domain import IncomingMessage
from vk_bot.transport.callback_api import VkCallbackTransport, _callback_to_message
from vk_bot.transport.long_polling import _event_to_message, _parse_longpoll_updates


class FakeEvent:
  def __init__(self, **kwargs) -> None:
    for k, v in kwargs.items():
      setattr(self, k, v)


@pytest.mark.asyncio
async def test_callback_confirmation():
  settings = VkBotSettings(
    vk_callback_confirmation="confirm123",
    vk_callback_path="/vk/callback",
  )
  transport = VkCallbackTransport(settings)
  async with AsyncClient(
    transport=ASGITransport(app=transport._app),
    base_url="http://test",
  ) as client:
    response = await client.post("/vk/callback", json={"type": "confirmation"})
  assert response.status_code == 200
  assert response.text == "confirm123"


@pytest.mark.asyncio
async def test_callback_invalid_secret():
  settings = VkBotSettings(
    vk_callback_secret="topsecret",
    vk_callback_path="/vk/callback",
  )
  transport = VkCallbackTransport(settings)
  async with AsyncClient(
    transport=ASGITransport(app=transport._app),
    base_url="http://test",
  ) as client:
    response = await client.post(
      "/vk/callback",
      json={"type": "message_new", "secret": "wrong", "object": {}},
    )
  assert response.status_code == 403


@pytest.mark.asyncio
async def test_callback_message_new_dispatches_handler():
  settings = VkBotSettings(vk_callback_path="/vk/callback")
  transport = VkCallbackTransport(settings)
  received: list[IncomingMessage] = []

  async def handler(message: IncomingMessage) -> None:
    received.append(message)

  transport._handler = handler
  payload = {
    "type": "message_new",
    "object": {
      "message": {
        "id": 1001,
        "conversation_message_id": 501,
        "from_id": 42,
        "peer_id": 42,
        "text": "Hello world question",
      }
    },
  }
  async with AsyncClient(
    transport=ASGITransport(app=transport._app),
    base_url="http://test",
  ) as client:
    response = await client.post("/vk/callback", json=payload)
  assert response.status_code == 200
  assert response.text == "ok"
  for _ in range(20):
    if received:
      break
    await asyncio.sleep(0.05)
  assert len(received) == 1
  assert received[0].text == "Hello world question"
  assert received[0].message_id == 1001
  assert received[0].conversation_message_id == 501


def test_event_to_message_with_from_me_outbox():
  event = FakeEvent(
    type=VkEventType.MESSAGE_NEW,
    user_id=1,
    peer_id=1,
    from_id=1,
    from_me=True,
    out=0,
    text="bot reply",
  )
  message, raw = _event_to_message(event)
  assert message is None
  assert raw is None


def test_event_to_message_with_attachments():
  event = FakeEvent(
    type=VkEventType.MESSAGE_NEW,
    user_id=1,
    peer_id=1,
    from_id=1,
    from_me=False,
    out=0,
    text="",
    message_id=77,
    attachments=[
      {
        "type": "doc",
        "doc": {"title": "doc", "ext": "pdf", "url": "http://x", "size": 10},
      }
    ],
  )

  message, raw = _event_to_message(event)
  assert message is not None
  assert message.from_id == 1
  assert message.is_outgoing is False
  assert message.message_id == 77
  assert isinstance(raw, list)
  assert len(raw) == 1


def test_event_to_message_outgoing_returns_none():
  event = FakeEvent(
    type=VkEventType.MESSAGE_NEW,
    user_id=1,
    peer_id=1,
    from_id=-1,
    out=1,
    text="bot reply",
  )
  message, raw = _event_to_message(event)
  assert message is None
  assert raw is None


def test_callback_to_message_outgoing_returns_none():
  payload = {
    "object": {
      "message": {
        "from_id": -1,
        "peer_id": 1,
        "out": 1,
        "text": "bot reply",
      }
    }
  }
  message, raw = _callback_to_message(payload)
  assert message is None
  assert raw is None


def test_callback_to_message_incoming_fields():
  payload = {
    "object": {
      "message": {
        "id": 200,
        "conversation_message_id": 300,
        "from_id": 42,
        "peer_id": 42,
        "out": 0,
        "text": "hello",
      }
    }
  }
  message, raw = _callback_to_message(payload)
  assert message is not None
  assert message.from_id == 42
  assert message.is_outgoing is False
  assert message.text == "hello"
  assert message.message_id == 200
  assert message.conversation_message_id == 300


@pytest.mark.asyncio
async def test_callback_outgoing_does_not_dispatch_handler():
  settings = VkBotSettings(vk_callback_path="/vk/callback")
  transport = VkCallbackTransport(settings)
  received: list[IncomingMessage] = []

  async def handler(message: IncomingMessage) -> None:
    received.append(message)

  transport._handler = handler
  payload = {
    "type": "message_new",
    "object": {
      "message": {
        "from_id": -1,
        "peer_id": 1,
        "out": 1,
        "text": "bot reply",
      }
    },
  }
  async with AsyncClient(
    transport=ASGITransport(app=transport._app),
    base_url="http://test",
  ) as client:
    response = await client.post("/vk/callback", json=payload)
  assert response.status_code == 200
  await asyncio.sleep(0.1)
  assert received == []


def test_parse_longpoll_updates_skips_malformed_events():
  class BrokenLongPoll:
    def _parse_event(self, raw_event):
      if raw_event == "bad":
        raise AttributeError("'Event' object has no attribute 'text'")
      return FakeEvent(
        type=VkEventType.MESSAGE_NEW,
        user_id=1,
        peer_id=1,
        from_id=2,
        out=0,
        text=str(raw_event),
      )

  longpoll = BrokenLongPoll()
  events = _parse_longpoll_updates(longpoll, ["ok", "bad", "after"])
  assert len(events) == 2
  assert events[0].text == "ok"
  assert events[1].text == "after"
