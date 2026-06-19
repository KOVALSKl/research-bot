import json

import fakeredis.aioredis
import pytest

from vk_bot.config import VkBotSettings
from vk_bot.state.user_queue import UserUploadQueueStore


@pytest.fixture
def settings() -> VkBotSettings:
  return VkBotSettings(vk_bot_redis_key_prefix="vk_bot:test:")


@pytest.fixture
async def store(settings):
  redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
  return UserUploadQueueStore(settings, redis_client), redis_client


@pytest.mark.asyncio
async def test_is_busy_when_batch_pending(store):
  queue, _ = store
  await queue.save_batch(1, task_ids=["t1"], filenames=["a.pdf"], peer_id=1)
  assert await queue.is_busy(1)


@pytest.mark.asyncio
async def test_clear_removes_batch(store):
  queue, redis_client = store
  await queue.save_batch(1, task_ids=["t1"], filenames=["a.pdf"], peer_id=1)
  await queue.clear(1)
  assert await queue.get_batch(1) is None
  assert not await queue.is_busy(1)


@pytest.mark.asyncio
async def test_list_active(store):
  queue, _ = store
  await queue.save_batch(2, task_ids=["t1"], filenames=["b.pdf"], peer_id=2)
  active = await queue.list_active()
  assert any(user_id == 2 for user_id, _ in active)
