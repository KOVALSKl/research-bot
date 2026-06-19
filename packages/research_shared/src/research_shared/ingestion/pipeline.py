from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from research_shared.config.settings import Settings
from research_shared.domain.models import DocumentRecord, IngestStatus
from research_shared.ingestion.file_storage import FileStorage
from research_shared.ingestion.protocols import Chunker, IngestionStateStore, PdfParser
from research_shared.storage.protocols import VectorStore


@dataclass
class IngestionResult:
    """Outcome of processing a single source document."""

    filename: str
    research_id: str
    status: IngestStatus
    chunk_count: int
    skipped: bool = False
    error: str | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


class IngestionPipeline:
    """Orchestrates parse → chunk → embed → upsert → state for a single PDF.

    Single entry point reused by the synchronous API path and Celery tasks.
    Idempotency is keyed on the deterministic ``content_hash``: re-processing an
    unchanged, already-indexed file is a no-op; a changed file removes the old
    chunks (by ``research_id``) and re-indexes.
    """

    def __init__(
        self,
        parser: PdfParser,
        chunker: Chunker,
        vector_store: VectorStore,
        state_store: IngestionStateStore,
        file_storage: FileStorage | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._parser = parser
        self._chunker = chunker
        self._vector_store = vector_store
        self._state_store = state_store
        self._file_storage = file_storage or FileStorage(self._settings)

    async def process(
        self,
        path: str | Path,
        display_name: str | None = None,
    ) -> IngestionResult:
        path = Path(path)
        stored = self._file_storage.describe(path)
        filename = stored.filename
        research_id = stored.research_id
        resolved_display_name = display_name or filename

        await self._state_store.ensure_collection()
        existing = await self._state_store.get(filename)

        if (
            existing is not None
            and existing.content_hash == stored.content_hash
            and existing.status == IngestStatus.INDEXED
        ):
            return IngestionResult(
                filename=filename,
                research_id=research_id,
                status=IngestStatus.INDEXED,
                chunk_count=existing.chunk_count,
                skipped=True,
            )

        await self._state_store.upsert(
            DocumentRecord(
                filename=filename,
                content_hash=stored.content_hash,
                research_id=research_id,
                display_name=resolved_display_name,
                status=IngestStatus.PROCESSING,
                chunk_count=0,
                indexed_at=existing.indexed_at if existing else None,
                updated_at=_now(),
            )
        )

        try:
            await self._vector_store.delete_by_research_id(research_id)
            if existing is not None and existing.research_id != research_id:
                await self._vector_store.delete_by_research_id(existing.research_id)

            document = self._parser.parse(path)
            chunks = self._chunker.chunk(document, research_id)
            chunks = [
                chunk.model_copy(update={"display_name": resolved_display_name})
                for chunk in chunks
            ]
            chunk_count = await self._vector_store.upsert(chunks)
        except Exception as exc:  # noqa: BLE001 — record failure durably, then re-raise
            await self._state_store.upsert(
                DocumentRecord(
                    filename=filename,
                    content_hash=stored.content_hash,
                    research_id=research_id,
                    display_name=resolved_display_name,
                    status=IngestStatus.FAILED,
                    chunk_count=0,
                    indexed_at=existing.indexed_at if existing else None,
                    updated_at=_now(),
                    error=str(exc),
                )
            )
            raise

        await self._state_store.upsert(
            DocumentRecord(
                filename=filename,
                content_hash=stored.content_hash,
                research_id=research_id,
                display_name=resolved_display_name,
                status=IngestStatus.INDEXED,
                chunk_count=chunk_count,
                indexed_at=_now(),
                updated_at=_now(),
            )
        )

        return IngestionResult(
            filename=filename,
            research_id=research_id,
            status=IngestStatus.INDEXED,
            chunk_count=chunk_count,
        )
