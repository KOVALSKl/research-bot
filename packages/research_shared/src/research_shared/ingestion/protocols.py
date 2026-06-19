from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from research_shared.domain.models import DocumentRecord, ResearchChunk


@dataclass
class ParsedPage:
    """A single parsed page: 1-based page number and its extracted text."""

    page: int
    text: str
    chapter: str | None = None


@dataclass
class ParsedDocument:
    """Result of parsing a PDF: title, pages with text, and source metadata."""

    title: str
    pages: list[ParsedPage]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PdfParser(Protocol):
    def parse(self, path: str | Path) -> ParsedDocument: ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, document: ParsedDocument, research_id: str) -> list[ResearchChunk]: ...


@runtime_checkable
class IngestionStateStore(Protocol):
    async def ensure_collection(self) -> None: ...

    async def get(self, filename: str) -> DocumentRecord | None: ...

    async def upsert(self, record: DocumentRecord) -> None: ...

    async def list(self) -> list[DocumentRecord]: ...
