from __future__ import annotations

from typing import Protocol

import redis.asyncio as aioredis

from vk_bot.config import VkBotSettings
from vk_bot.domain import IncomingMessage


class MessageDedupStore(Protocol):
  async def try_acquire(self, message: IncomingMessage) -> bool: ...


class MemoryMessageDedupStore:
  def __init__(self) -> None:
    self._seen: set[str] = set()

  async def try_acquire(self, message: IncomingMessage) -> bool:
    key = _dedup_key("", message)
    if key in self._seen:
      return False
    self._seen.add(key)
    return True


class RedisMessageDedupStore:
  def __init__(self, settings: VkBotSettings, redis_client: aioredis.Redis) -> None:
    self._settings = settings
    self._redis = redis_client

  async def try_acquire(self, message: IncomingMessage) -> bool:
    key = _dedup_key(self._settings.redis_key_prefix, message)
    result = await self._redis.set(
      key,
      "1",
      nx=True,
      ex=self._settings.vk_message_dedup_ttl_seconds,
    )
    return bool(result)


def _dedup_key(prefix: str, message: IncomingMessage) -> str:
  if message.message_id:
    return f"{prefix}dedup:{message.message_id}"
  if message.conversation_message_id:
    return f"{prefix}dedup:cmid:{message.peer_id}:{message.conversation_message_id}"
  return f"{prefix}dedup:fallback:{message.peer_id}:{message.user_id}:{hash(message.text)}"


def create_message_dedup_store(
  settings: VkBotSettings,
  redis_client: aioredis.Redis | None = None,
) -> MessageDedupStore:
  if redis_client is None:
    return MemoryMessageDedupStore()
  return RedisMessageDedupStore(settings, redis_client)
