from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import redis.asyncio as aioredis

from vk_bot.config import VkBotSettings
from vk_bot.core_api.client import CoreApiClient
from vk_bot.handlers import messages as ux
from vk_bot.vk.api import VkApiClientProtocol

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"indexed", "failed"})


@dataclass
class UserBatch:
  task_ids: list[str]
  filenames: list[str]
  status: str = "pending"
  created_at: str = ""
  peer_id: int = 0


class UserUploadQueueStore:
  def __init__(self, settings: VkBotSettings, redis_client: aioredis.Redis) -> None:
    self._settings = settings
    self._redis = redis_client

  def _key(self, user_id: int) -> str:
    return f"{self._settings.redis_key_prefix}user:{user_id}:batch"

  async def is_busy(self, user_id: int) -> bool:
    batch = await self.get_batch(user_id)
    return batch is not None and batch.status == "pending"

  async def save_batch(
    self,
    user_id: int,
    *,
    task_ids: list[str],
    filenames: list[str],
    peer_id: int,
  ) -> None:
    if await self.is_busy(user_id):
      raise RuntimeError("User upload queue is busy")

    payload = UserBatch(
      task_ids=task_ids,
      filenames=filenames,
      status="pending",
      created_at=datetime.now(UTC).isoformat(),
      peer_id=peer_id,
    )
    await self._redis.set(
      self._key(user_id),
      json.dumps(payload.__dict__),
      ex=self._settings.vk_user_batch_ttl_seconds,
    )

  async def get_batch(self, user_id: int) -> UserBatch | None:
    raw = await self._redis.get(self._key(user_id))
    if raw is None:
      return None
    data = json.loads(raw)
    return UserBatch(
      task_ids=list(data.get("task_ids", [])),
      filenames=list(data.get("filenames", [])),
      status=str(data.get("status", "pending")),
      created_at=str(data.get("created_at", "")),
      peer_id=int(data.get("peer_id", 0)),
    )

  async def clear(self, user_id: int) -> None:
    await self._redis.delete(self._key(user_id))

  async def list_active(self) -> list[tuple[int, UserBatch]]:
    pattern = f"{self._settings.redis_key_prefix}user:*:batch"
    active: list[tuple[int, UserBatch]] = []
    async for key in self._redis.scan_iter(match=pattern):
      key_str = key.decode() if isinstance(key, bytes) else str(key)
      parts = key_str.split(":")
      if len(parts) < 3:
        continue
      try:
        user_id = int(parts[-2])
      except ValueError:
        continue
      batch = await self.get_batch(user_id)
      if batch is not None and batch.status == "pending":
        active.append((user_id, batch))
    return active


class BatchPollerProtocol(Protocol):
  async def start(self) -> None: ...

  async def stop(self) -> None: ...


class BatchPoller:
  def __init__(
    self,
    settings: VkBotSettings,
    queue_store: UserUploadQueueStore,
    core_api: CoreApiClient,
    vk_api: VkApiClientProtocol,
  ) -> None:
    self._settings = settings
    self._queue = queue_store
    self._core_api = core_api
    self._vk_api = vk_api
    self._task: asyncio.Task[None] | None = None
    self._stop = asyncio.Event()

  async def start(self) -> None:
    self._stop.clear()
    self._task = asyncio.create_task(self._loop())

  async def stop(self) -> None:
    self._stop.set()
    if self._task is not None:
      self._task.cancel()
      try:
        await self._task
      except asyncio.CancelledError:
        pass

  async def _loop(self) -> None:
    while not self._stop.is_set():
      try:
        active = await self._queue.list_active()
        for user_id, batch in active:
          await self._check_batch(user_id, batch)
      except Exception:
        logger.exception("Batch poller iteration failed")
      try:
        await asyncio.wait_for(
          self._stop.wait(),
          timeout=self._settings.vk_batch_poll_interval_seconds,
        )
        break
      except TimeoutError:
        continue

  async def _check_batch(self, user_id: int, batch: UserBatch) -> None:
    statuses: list[dict[str, Any]] = []
    for task_id in batch.task_ids:
      statuses.append(await self._core_api.get_task_status(task_id))

    if not all(s.get("status") in TERMINAL_STATUSES for s in statuses):
      return

    indexed = sum(1 for s in statuses if s.get("status") == "indexed")
    failed_errors = [
      str(s.get("error") or s.get("detail") or "unknown error")
      for s in statuses
      if s.get("status") == "failed"
    ]
    peer_id = batch.peer_id or user_id
    await self._vk_api.send_message(
      peer_id,
      ux.batch_completed(
        indexed=indexed,
        total=len(batch.task_ids),
        failed_errors=failed_errors or None,
      ),
    )
    await self._queue.clear(user_id)
