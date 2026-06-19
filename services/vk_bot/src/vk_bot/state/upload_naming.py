from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import redis.asyncio as aioredis

from vk_bot.config import VkBotSettings


@dataclass
class NamingFile:
  original_name: str
  temp_path: str
  size: int = 0


@dataclass
class UploadNamingSessionData:
  peer_id: int
  files: list[NamingFile] = field(default_factory=list)
  names: list[str] = field(default_factory=list)
  current_index: int = 0
  created_at: str = ""


class UploadNamingStore(Protocol):
  async def active(self, user_id: int) -> bool: ...

  async def get(self, user_id: int) -> UploadNamingSessionData | None: ...

  def save_temp_file(self, user_id: int, content: bytes, original_name: str) -> NamingFile: ...

  async def save(
    self,
    user_id: int,
    *,
    peer_id: int,
    files: list[NamingFile],
  ) -> None: ...

  async def append_name(self, user_id: int, name: str) -> UploadNamingSessionData | None: ...

  async def advance(self, user_id: int) -> UploadNamingSessionData | None: ...

  async def clear(self, user_id: int) -> None: ...


class MemoryUploadNamingStore:
  def __init__(self, settings: VkBotSettings) -> None:
    self._settings = settings
    self._sessions: dict[int, UploadNamingSessionData] = {}

  def _temp_dir(self, user_id: int) -> Path:
    base = Path(self._settings.vk_naming_temp_dir)
    path = base / str(user_id)
    path.mkdir(parents=True, exist_ok=True)
    return path

  def save_temp_file(self, user_id: int, content: bytes, original_name: str) -> NamingFile:
    temp_dir = self._temp_dir(user_id)
    temp_path = temp_dir / f"{uuid.uuid4().hex}.pdf"
    temp_path.write_bytes(content)
    return NamingFile(
      original_name=original_name,
      temp_path=str(temp_path),
      size=len(content),
    )

  async def active(self, user_id: int) -> bool:
    return user_id in self._sessions

  async def get(self, user_id: int) -> UploadNamingSessionData | None:
    return self._sessions.get(user_id)

  async def save(
    self,
    user_id: int,
    *,
    peer_id: int,
    files: list[NamingFile],
  ) -> None:
    self._sessions[user_id] = UploadNamingSessionData(
      peer_id=peer_id,
      files=files,
      names=[],
      current_index=0,
      created_at=datetime.now(UTC).isoformat(),
    )

  async def append_name(self, user_id: int, name: str) -> UploadNamingSessionData | None:
    session = self._sessions.get(user_id)
    if session is None:
      return None
    session.names.append(name)
    return session

  async def advance(self, user_id: int) -> UploadNamingSessionData | None:
    session = self._sessions.get(user_id)
    if session is None:
      return None
    session.current_index += 1
    return session

  async def clear(self, user_id: int) -> None:
    session = self._sessions.pop(user_id, None)
    if session is not None:
      _remove_temp_files(session)


class RedisUploadNamingStore:
  def __init__(self, settings: VkBotSettings, redis_client: aioredis.Redis) -> None:
    self._settings = settings
    self._redis = redis_client

  def _key(self, user_id: int) -> str:
    return f"{self._settings.redis_key_prefix}user:{user_id}:naming"

  def _temp_dir(self, user_id: int) -> Path:
    base = Path(self._settings.vk_naming_temp_dir)
    path = base / str(user_id)
    path.mkdir(parents=True, exist_ok=True)
    return path

  def save_temp_file(self, user_id: int, content: bytes, original_name: str) -> NamingFile:
    temp_dir = self._temp_dir(user_id)
    temp_path = temp_dir / f"{uuid.uuid4().hex}.pdf"
    temp_path.write_bytes(content)
    return NamingFile(
      original_name=original_name,
      temp_path=str(temp_path),
      size=len(content),
    )

  async def active(self, user_id: int) -> bool:
    return bool(await self._redis.exists(self._key(user_id)))

  async def get(self, user_id: int) -> UploadNamingSessionData | None:
    raw = await self._redis.get(self._key(user_id))
    if raw is None:
      return None
    return _deserialize_session(raw)

  async def save(
    self,
    user_id: int,
    *,
    peer_id: int,
    files: list[NamingFile],
  ) -> None:
    session = UploadNamingSessionData(
      peer_id=peer_id,
      files=files,
      names=[],
      current_index=0,
      created_at=datetime.now(UTC).isoformat(),
    )
    await self._redis.set(
      self._key(user_id),
      _serialize_session(session),
      ex=self._settings.vk_naming_session_ttl_seconds,
    )

  async def _persist(self, user_id: int, session: UploadNamingSessionData) -> None:
    await self._redis.set(
      self._key(user_id),
      _serialize_session(session),
      ex=self._settings.vk_naming_session_ttl_seconds,
    )

  async def append_name(self, user_id: int, name: str) -> UploadNamingSessionData | None:
    session = await self.get(user_id)
    if session is None:
      return None
    session.names.append(name)
    await self._persist(user_id, session)
    return session

  async def advance(self, user_id: int) -> UploadNamingSessionData | None:
    session = await self.get(user_id)
    if session is None:
      return None
    session.current_index += 1
    await self._persist(user_id, session)
    return session

  async def clear(self, user_id: int) -> None:
    session = await self.get(user_id)
    await self._redis.delete(self._key(user_id))
    if session is not None:
      _remove_temp_files(session)


def _serialize_session(session: UploadNamingSessionData) -> str:
  payload = {
    "peer_id": session.peer_id,
    "files": [asdict(file) for file in session.files],
    "names": session.names,
    "current_index": session.current_index,
    "created_at": session.created_at,
  }
  return json.dumps(payload)


def _deserialize_session(raw: str) -> UploadNamingSessionData:
  data = json.loads(raw)
  files = [
    NamingFile(
      original_name=str(item.get("original_name", "")),
      temp_path=str(item.get("temp_path", "")),
      size=int(item.get("size", 0)),
    )
    for item in data.get("files", [])
  ]
  return UploadNamingSessionData(
    peer_id=int(data.get("peer_id", 0)),
    files=files,
    names=list(data.get("names", [])),
    current_index=int(data.get("current_index", 0)),
    created_at=str(data.get("created_at", "")),
  )


def _remove_temp_files(session: UploadNamingSessionData) -> None:
  for file in session.files:
    path = Path(file.temp_path)
    if path.is_file():
      path.unlink(missing_ok=True)


def create_upload_naming_store(
  settings: VkBotSettings,
  redis_client: aioredis.Redis | None = None,
) -> MemoryUploadNamingStore | RedisUploadNamingStore:
  if redis_client is None:
    return MemoryUploadNamingStore(settings)
  return RedisUploadNamingStore(settings, redis_client)
