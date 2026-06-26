from __future__ import annotations

from research_shared.ingestion.yandex_disk import YandexDiskStorage
from research_shared.literature.storage.local_cache import LocalPdfCache


class YandexDiskPdfCache:
    """External PDF cache on Yandex Disk under ``{base_path}/external/``."""

    def __init__(self, storage: YandexDiskStorage, *, subfolder: str = "external") -> None:
        self._storage = storage
        self._prefix = subfolder.strip("/")

    def _filename(self, cache_key: str) -> str:
        safe_key = cache_key.replace("/", "_")
        return f"{self._prefix}/{safe_key}.pdf"

    def exists(self, cache_key: str) -> bool:
        try:
            self._storage.describe(self._filename(cache_key))
            return True
        except FileNotFoundError:
            return False

    def read(self, cache_key: str) -> bytes | None:
        try:
            return self._storage.read(self._filename(cache_key))
        except FileNotFoundError:
            return None

    def write(self, cache_key: str, content: bytes) -> None:
        self._storage.ensure_folder(f"{self._storage.base_path}/{self._prefix}")
        self._storage.save(self._filename(cache_key), content)


def create_pdf_cache_storage(settings) -> LocalPdfCache | YandexDiskPdfCache:
    from research_shared.config.settings import Settings
    from research_shared.ingestion.factory import create_document_storage

    if not isinstance(settings, Settings):
        settings = Settings()

    if settings.storage_backend == "yandex" and settings.yandex_disk_api_token.strip():
        doc_storage = create_document_storage(settings)
        if isinstance(doc_storage, YandexDiskStorage):
            return YandexDiskPdfCache(doc_storage)
    return LocalPdfCache(settings.external_pdf_cache_dir)
