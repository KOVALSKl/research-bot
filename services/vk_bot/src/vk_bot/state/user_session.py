from __future__ import annotations

import time
from typing import Protocol

import redis.asyncio as aioredis

from vk_bot.config import VkBotSettings


class UserSessionStore(Protocol):
  async def has_seen(self, user_id: int) -> bool: ...

  async def mark_seen(self, user_id: int) -> None: ...


class MemoryUserSessionStore:
  def __init__(self) -> None:
    self._seen: set[int] = set()

  async def has_seen(self, user_id: int) -> bool:
    return user_id in self._seen

  async def mark_seen(self, user_id: int) -> None:
    self._seen.add(user_id)


class RedisUserSessionStore:
  def __init__(self, settings: VkBotSettings, redis_client: aioredis.Redis) -> None:
    self._settings = settings
    self._redis = redis_client

  def _key(self, user_id: int) -> str:
    return f"{self._settings.redis_key_prefix}user:{user_id}:seen"

  async def has_seen(self, user_id: int) -> bool:
    return bool(await self._redis.exists(self._key(user_id)))

  async def mark_seen(self, user_id: int) -> None:
    await self._redis.set(self._key(user_id), str(time.time()))


def create_user_session_store(
  settings: VkBotSettings,
  redis_client: aioredis.Redis | None = None,
) -> UserSessionStore:
  if redis_client is None:
    return MemoryUserSessionStore()
  return RedisUserSessionStore(settings, redis_client)
