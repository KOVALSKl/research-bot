from dataclasses import dataclass
from datetime import datetime, timezone
import tempfile
from pathlib import Path

from research_shared.config.settings import Settings
from research_shared.domain.models import DocumentRecord, IngestStatus
from research_shared.ingestion.factory import create_archive_storage, create_staging_storage
from research_shared.ingestion.file_storage import compute_content_hash, compute_research_id
from research_shared.ingestion.protocols import Chunker, IngestionStateStore, PdfParser
from research_shared.ingestion.staging_storage import StagingStorage
from research_shared.ingestion.storage_protocol import DocumentStorage
from research_shared.ingestion.yandex_disk import YandexDiskStorage
from research_shared.logging_config import get_logger
from research_shared.storage.protocols import VectorStore

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Outcome of processing a single source document."""

    filename: str
    research_id: str
    status: IngestStatus
    chunk_count: int
    skipped: bool = False
    error: str | None = None
    archive_error: str | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


class IngestionPipeline:
    """Orchestrates staging read → parse → chunk → embed → upsert → archive.

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
        staging_storage: StagingStorage | None = None,
        archive_storage: DocumentStorage | None = None,
        file_storage: DocumentStorage | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._parser = parser
        self._chunker = chunker
        self._vector_store = vector_store
        self._state_store = state_store
        self._staging_storage = staging_storage or create_staging_storage(self._settings)
        archive = archive_storage or file_storage
        self._archive_storage = archive or create_archive_storage(self._settings)

    async def process(
        self,
        filename: str,
        display_name: str | None = None,
    ) -> IngestionResult:
        safe_name = Path(filename).name
        content = self._staging_storage.read(safe_name)
        content_hash = compute_content_hash(content)
        research_id = compute_research_id(content)
        resolved_display_name = (display_name or "").strip() or safe_name

        await self._state_store.ensure_collection()
        existing = await self._state_store.get(safe_name)

        if (
            existing is not None
            and existing.content_hash == content_hash
            and existing.status == IngestStatus.INDEXED
        ):
            self._staging_storage.delete(safe_name)
            return IngestionResult(
                filename=safe_name,
                research_id=research_id,
                status=IngestStatus.INDEXED,
                chunk_count=existing.chunk_count,
                skipped=True,
            )

        await self._state_store.upsert(
            DocumentRecord(
                filename=safe_name,
                content_hash=content_hash,
                research_id=research_id,
                display_name=resolved_display_name,
                status=IngestStatus.PROCESSING,
                chunk_count=0,
                indexed_at=existing.indexed_at if existing else None,
                updated_at=_now(),
                source_url=existing.source_url if existing else None,
                archive_path=existing.archive_path if existing else None,
                archive_error=None,
            )
        )

        try:
            await self._vector_store.delete_by_research_id(research_id)
            if existing is not None and existing.research_id != research_id:
                await self._vector_store.delete_by_research_id(existing.research_id)

            temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            temp_path = Path(temp_file.name)
            try:
                temp_file.write(content)
                temp_file.close()
                document = self._parser.parse(temp_path)
            finally:
                temp_path.unlink(missing_ok=True)

            chunks = self._chunker.chunk(document, research_id)
            chunks = [
                chunk.model_copy(update={"display_name": resolved_display_name, "source_path": None})
                for chunk in chunks
            ]
            chunk_count = await self._vector_store.upsert(chunks)
        except Exception as exc:  # noqa: BLE001 — record failure durably, then re-raise
            await self._state_store.upsert(
                DocumentRecord(
                    filename=safe_name,
                    content_hash=content_hash,
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

        source_url, archive_path, archive_error = await self._archive_after_index(
            safe_name,
            content,
            research_id=research_id,
            display_name=resolved_display_name,
            content_hash=content_hash,
            chunk_count=chunk_count,
            indexed_at=_now(),
        )

        if source_url:
            enriched_chunks = [
                c.model_copy(update={"metadata": {**c.metadata, "source_url": source_url}})
                for c in chunks
            ]
            await self._vector_store.upsert(enriched_chunks)

        if archive_error is None:
            self._staging_storage.delete(safe_name)

        return IngestionResult(
            filename=safe_name,
            research_id=research_id,
            status=IngestStatus.INDEXED,
            chunk_count=chunk_count,
            archive_error=archive_error,
        )

    async def archive_only(
        self,
        filename: str,
        *,
        display_name: str | None = None,
    ) -> IngestionResult:
        """Retry archive for an already-indexed document (staging must still exist)."""
        safe_name = Path(filename).name
        existing = await self._state_store.get(safe_name)
        if existing is None or existing.status != IngestStatus.INDEXED:
            raise ValueError(f"Document {safe_name} is not indexed")

        content = self._staging_storage.read(safe_name)
        resolved_display_name = display_name or existing.display_name or safe_name
        source_url, archive_path, archive_error = await self._archive_after_index(
            safe_name,
            content,
            research_id=existing.research_id,
            display_name=resolved_display_name,
            content_hash=existing.content_hash,
            chunk_count=existing.chunk_count,
            indexed_at=existing.indexed_at or _now(),
        )

        if archive_error is None:
            self._staging_storage.delete(safe_name)

        return IngestionResult(
            filename=safe_name,
            research_id=existing.research_id,
            status=IngestStatus.INDEXED,
            chunk_count=existing.chunk_count,
            archive_error=archive_error,
        )

    async def _archive_after_index(
        self,
        filename: str,
        content: bytes,
        *,
        research_id: str,
        display_name: str,
        content_hash: str,
        chunk_count: int,
        indexed_at: datetime,
    ) -> tuple[str | None, str | None, str | None]:
        source_url: str | None = None
        archive_path: str | None = None
        archive_error: str | None = None

        try:
            self._archive_storage.save(filename, content)
            if isinstance(self._archive_storage, YandexDiskStorage):
                archive_path = f"{self._archive_storage.base_path}/{Path(filename).name}"
                try:
                    source_url = self._archive_storage.publish_and_get_url(filename)
                except Exception as exc:  # noqa: BLE001 — publish failure must not fail index
                    logger.warning(
                        "Yandex Disk publish failed",
                        extra={
                            "filename": filename,
                            "research_id": research_id,
                            "event": "ingest.archive.publish_failed",
                        },
                        exc_info=True,
                    )
                    archive_error = str(exc)
            else:
                stored = self._archive_storage.describe(filename)
                archive_path = str(stored.path) if stored.path else filename
        except Exception as exc:  # noqa: BLE001 — keep document indexed, retry archive later
            archive_error = str(exc)
            logger.error(
                "Document archive failed after index",
                extra={
                    "filename": filename,
                    "research_id": research_id,
                    "event": "ingest.archive.failed",
                },
                exc_info=True,
            )

        await self._state_store.upsert(
            DocumentRecord(
                filename=filename,
                content_hash=content_hash,
                research_id=research_id,
                display_name=display_name,
                status=IngestStatus.INDEXED,
                chunk_count=chunk_count,
                indexed_at=indexed_at,
                updated_at=_now(),
                source_url=source_url,
                archive_path=archive_path,
                archive_error=archive_error,
            )
        )
        return source_url, archive_path, archive_error
