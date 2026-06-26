from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class StoredFile:
    filename: str
    content_hash: str
    research_id: str
    path: Path | None = None


@dataclass(frozen=True)
class PdfFileInfo:
    filename: str
    size: int
    modified: str | None = None
    content_hash: str | None = None


@runtime_checkable
class DocumentStorage(Protocol):
    def save(self, filename: str, content: bytes) -> StoredFile: ...

    def list_pdfs(self) -> list[PdfFileInfo]: ...

    def read(self, filename: str) -> bytes: ...

    def describe(self, filename: str) -> StoredFile: ...

    def delete(self, filename: str) -> None: ...
