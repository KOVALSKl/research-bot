from __future__ import annotations

from research_shared.config.settings import Settings
from research_shared.ingestion.file_storage import LocalDocumentStorage
from research_shared.ingestion.staging_storage import LocalStagingStorage, StagingStorage
from research_shared.ingestion.storage_protocol import DocumentStorage
from research_shared.ingestion.yandex_disk import YandexDiskStorage


def create_staging_storage(settings: Settings) -> StagingStorage:
    return LocalStagingStorage(settings)


def create_archive_storage(settings: Settings) -> DocumentStorage:
    if settings.storage_backend == "yandex" and settings.yandex_disk_api_token.strip():
        return YandexDiskStorage(
            token=settings.yandex_disk_api_token,
            base_path=settings.yandex_disk_base_path,
        )
    return LocalDocumentStorage(settings)


def create_document_storage(settings: Settings) -> DocumentStorage:
    """Backward-compatible alias for archive storage."""
    return create_archive_storage(settings)
