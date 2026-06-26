from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PdfCacheStorage(Protocol):
    def exists(self, cache_key: str) -> bool: ...

    def read(self, cache_key: str) -> bytes | None: ...

    def write(self, cache_key: str, content: bytes) -> None: ...
