from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from typing import Protocol

import redis.asyncio as aioredis

from vk_bot.config import VkBotSettings


class RateLimiter(Protocol):
  async def allow(
    self,
    user_id: int,
    text: str = "",
    *,
    attachment_count: int = 0,
  ) -> bool: ...

  async def should_notify_rate_limit(self, user_id: int) -> bool: ...

  async def mark_rate_limit_notified(self, user_id: int) -> None: ...


class MemoryRateLimiter:
  def __init__(self, settings: VkBotSettings) -> None:
    self._max_messages = settings.vk_rate_limit_max_messages
    self._window = settings.vk_rate_limit_window_seconds
    self._debounce = settings.vk_debounce_seconds
    self._hits: dict[int, list[float]] = defaultdict(list)
    self._debounce_keys: dict[str, float] = {}
    self._notify_times: dict[int, float] = {}

  def _debounce_key(self, user_id: int, text: str, attachment_count: int) -> str | None:
    if self._debounce <= 0:
      return None
    if text:
      digest = hashlib.sha256(text.encode()).hexdigest()[:16]
      return f"{user_id}:{digest}"
    if attachment_count > 0:
      return f"{user_id}:{attachment_count}"
    return None

  async def allow(
    self,
    user_id: int,
    text: str = "",
    *,
    attachment_count: int = 0,
  ) -> bool:
    now = time.monotonic()
    debounce_key = self._debounce_key(user_id, text, attachment_count)
    if debounce_key is not None:
      previous = self._debounce_keys.get(debounce_key)
      if previous is not None and now - previous < self._debounce:
        return False
      self._debounce_keys[debounce_key] = now

    window_start = now - self._window
    hits = [timestamp for timestamp in self._hits[user_id] if timestamp >= window_start]
    if len(hits) >= self._max_messages:
      self._hits[user_id] = hits
      return False
    hits.append(now)
    self._hits[user_id] = hits
    return True

  async def should_notify_rate_limit(self, user_id: int) -> bool:
    now = time.monotonic()
    last_notified = self._notify_times.get(user_id)
    if last_notified is not None and now - last_notified < self._window:
      return False
    return True

  async def mark_rate_limit_notified(self, user_id: int) -> None:
    self._notify_times[user_id] = time.monotonic()


class RedisRateLimiter:
  def __init__(self, settings: VkBotSettings, redis_client: aioredis.Redis) -> None:
    self._settings = settings
    self._redis = redis_client
    self._max_messages = settings.vk_rate_limit_max_messages
    self._window = settings.vk_rate_limit_window_seconds
    self._debounce = settings.vk_debounce_seconds

  def _rate_key(self, user_id: int) -> str:
    return f"{self._settings.redis_key_prefix}rate:{user_id}"

  def _debounce_key(self, user_id: int, text: str, attachment_count: int) -> str | None:
    if self._debounce <= 0:
      return None
    if text:
      digest = hashlib.sha256(text.encode()).hexdigest()[:16]
      return f"{self._settings.redis_key_prefix}debounce:{user_id}:{digest}"
    if attachment_count > 0:
      return f"{self._settings.redis_key_prefix}debounce:{user_id}:{attachment_count}"
    return None

  def _notify_key(self, user_id: int) -> str:
    return f"{self._settings.redis_key_prefix}rate_notify:{user_id}"

  async def allow(
    self,
    user_id: int,
    text: str = "",
    *,
    attachment_count: int = 0,
  ) -> bool:
    debounce_key = self._debounce_key(user_id, text, attachment_count)
    if debounce_key is not None:
      acquired = await self._redis.set(debounce_key, "1", nx=True, ex=self._debounce)
      if not acquired:
        return False

    rate_key = self._rate_key(user_id)
    pipe = self._redis.pipeline()
    now = time.time()
    window_start = now - self._window
    pipe.zremrangebyscore(rate_key, 0, window_start)
    pipe.zadd(rate_key, {str(now): now})
    pipe.zcard(rate_key)
    pipe.expire(rate_key, self._window)
    _, _, count, _ = await pipe.execute()
    return int(count) <= self._max_messages

  async def should_notify_rate_limit(self, user_id: int) -> bool:
    return not bool(await self._redis.exists(self._notify_key(user_id)))

  async def mark_rate_limit_notified(self, user_id: int) -> None:
    await self._redis.set(self._notify_key(user_id), "1", ex=self._window)


def create_rate_limiter(
  settings: VkBotSettings,
  redis_client: aioredis.Redis | None = None,
) -> RateLimiter:
  if settings.vk_rate_limit_backend == "memory":
    return MemoryRateLimiter(settings)
  if redis_client is None:
    raise ValueError("Redis client is required for redis rate limit backend")
  return RedisRateLimiter(settings, redis_client)
