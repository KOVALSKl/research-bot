"""Per-user conversation history stored in Redis."""

from __future__ import annotations

import json

from redis.asyncio import Redis

from research_shared.logging_config import get_logger

logger = get_logger(__name__)

ConversationTurn = dict[str, str]  # {"role": "user"|"assistant", "content": "..."}


class ConversationStore:
    """Stores the last N question/answer turns per user in Redis with TTL."""

    def __init__(
        self,
        redis: Redis,
        prefix: str,
        max_turns: int = 5,
        ttl_seconds: int = 3600,
    ) -> None:
        self._redis = redis
        self._prefix = prefix
        self._max_turns = max_turns
        self._ttl = ttl_seconds

    def _key(self, user_id: int) -> str:
        return f"{self._prefix}conversation:{user_id}"

    async def get(self, user_id: int) -> list[ConversationTurn]:
        raw = await self._redis.get(self._key(user_id))
        if not raw:
            return []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Invalid conversation history in Redis",
                extra={"user_id": user_id, "event": "conversation.parse_error"},
            )
        return []

    async def append(self, user_id: int, question: str, answer: str) -> None:
        history = await self.get(user_id)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        # Keep only the last max_turns pairs (2 messages per turn)
        max_messages = self._max_turns * 2
        if len(history) > max_messages:
            history = history[-max_messages:]
        key = self._key(user_id)
        await self._redis.set(key, json.dumps(history, ensure_ascii=False), ex=self._ttl)

    async def clear(self, user_id: int) -> None:
        await self._redis.delete(self._key(user_id))
