import pytest

from vk_bot.domain import IncomingMessage
from vk_bot.state.message_dedup import MemoryMessageDedupStore


@pytest.mark.asyncio
async def test_dedup_acquires_once():
  store = MemoryMessageDedupStore()
  message = IncomingMessage(user_id=1, peer_id=1, message_id=123)

  assert await store.try_acquire(message) is True
  assert await store.try_acquire(message) is False


@pytest.mark.asyncio
async def test_dedup_uses_conversation_message_id():
  store = MemoryMessageDedupStore()
  message = IncomingMessage(
    user_id=1,
    peer_id=100,
    conversation_message_id=55,
  )

  assert await store.try_acquire(message) is True
  assert await store.try_acquire(message) is False
