from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from research_shared.config.settings import Settings
from research_shared.domain.models import DocumentRecord, IngestStatus
from research_shared.ingestion.file_storage import FileStorage
from research_shared.ingestion.staging_storage import LocalStagingStorage

from core_api.api.routes import documents
from core_api.dependencies import get_app_state


def test_upload_document_returns_202(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(documents.router, prefix="/documents")

    staging_dir = tmp_path / "staging"
    settings = Settings(
        researches_dir=str(tmp_path / "archive"),
        ingest_staging_dir=str(staging_dir),
        ingest_sync=False,
    )
    staging_storage = LocalStagingStorage(settings)

    class FakeCelery:
        def enqueue_index_document(self, path: str, *, display_name: str | None = None) -> str:
            return "task-123"

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        settings=settings,
        staging_storage=staging_storage,
        archive_storage=MagicMock(),
        celery_client=FakeCelery(),
        ingestion_pipeline=None,
    )

    client = TestClient(app)
    response = client.post(
        "/documents",
        files={"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["task_id"] == "task-123"
    assert data["filename"] == "paper.pdf"
    assert (staging_dir / "paper.pdf").exists()


def test_upload_document_yandex_backend_uses_staging_only(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(documents.router, prefix="/documents")

    staging_dir = tmp_path / "staging"
    settings = Settings(
        _env_file=None,
        storage_backend="yandex",
        yandex_disk_api_token="token",
        ingest_staging_dir=str(staging_dir),
        ingest_sync=False,
    )
    staging_storage = LocalStagingStorage(settings)
    archive_storage = MagicMock()
    archive_storage.save.side_effect = AssertionError("Yandex must not be called on upload")

    class FakeCelery:
        def enqueue_index_document(self, path: str, *, display_name: str | None = None) -> str:
            return "task-yandex"

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        settings=settings,
        staging_storage=staging_storage,
        archive_storage=archive_storage,
        celery_client=FakeCelery(),
        ingestion_pipeline=None,
    )

    client = TestClient(app)
    response = client.post(
        "/documents",
        files={"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 202
    archive_storage.save.assert_not_called()
    assert (staging_dir / "paper.pdf").exists()


def test_download_source_file_by_research_id(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(documents.router, prefix="/documents")

    settings = Settings(researches_dir=str(tmp_path), ingest_sync=False)
    archive_storage = FileStorage(settings)
    stored = archive_storage.save("paper.pdf", b"%PDF-1.4 content")

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        settings=settings,
        archive_storage=archive_storage,
        ingestion_pipeline=None,
    )

    client = TestClient(app)
    response = client.get(f"/documents/files/{stored.research_id}")

    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 content"
    assert response.headers["content-type"] == "application/pdf"


def test_download_source_file_redirects_to_source_url(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(documents.router, prefix="/documents")

    settings = Settings(researches_dir=str(tmp_path), ingest_sync=False)

    class FakeStateStore:
        async def list(self) -> list[DocumentRecord]:
            return [
                DocumentRecord(
                    filename="paper.pdf",
                    content_hash="abc",
                    research_id="r1",
                    status=IngestStatus.INDEXED,
                    source_url="https://disk.yandex.ru/i/example",
                )
            ]

    class FakePipeline:
        _state_store = FakeStateStore()

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        settings=settings,
        archive_storage=MagicMock(),
        ingestion_pipeline=FakePipeline(),
    )

    client = TestClient(app, follow_redirects=False)
    response = client.get("/documents/files/r1")

    assert response.status_code == 302
    assert response.headers["location"] == "https://disk.yandex.ru/i/example"


def test_download_source_file_unknown_research_id(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(documents.router, prefix="/documents")

    settings = Settings(researches_dir=str(tmp_path), ingest_sync=False)
    archive_storage = FileStorage(settings)

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        settings=settings,
        archive_storage=archive_storage,
        ingestion_pipeline=None,
    )

    client = TestClient(app)
    response = client.get("/documents/files/unknown-id")

    assert response.status_code == 404


def test_list_documents_empty(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(documents.router, prefix="/documents")

    settings = Settings(researches_dir=str(tmp_path), ingest_sync=False)

    class FakeStateStore:
        async def list(self) -> list[DocumentRecord]:
            return []

    class FakePipeline:
        _state_store = FakeStateStore()

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        settings=settings,
        archive_storage=FileStorage(settings),
        ingestion_pipeline=FakePipeline(),
    )

    client = TestClient(app)
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {"documents": []}


def test_list_documents_with_filter(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(documents.router, prefix="/documents")

    settings = Settings(researches_dir=str(tmp_path), ingest_sync=False)
    indexed = DocumentRecord(
        filename="paper.pdf",
        content_hash="abc",
        research_id="r1",
        display_name="My Paper",
        status=IngestStatus.INDEXED,
        chunk_count=10,
        indexed_at=datetime.now(UTC),
        source_url="https://disk.yandex.ru/i/example",
    )
    queued = DocumentRecord(
        filename="other.pdf",
        content_hash="def",
        research_id="r2",
        status=IngestStatus.QUEUED,
    )

    class FakeStateStore:
        async def list(self) -> list[DocumentRecord]:
            return [indexed, queued]

    class FakePipeline:
        _state_store = FakeStateStore()

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        settings=settings,
        archive_storage=FileStorage(settings),
        ingestion_pipeline=FakePipeline(),
    )

    client = TestClient(app)
    response = client.get("/documents", params={"status": "indexed"})

    assert response.status_code == 200
    data = response.json()["documents"]
    assert len(data) == 1
    assert data[0]["display_name"] == "My Paper"
    assert data[0]["chunk_count"] == 10
    assert data[0]["source_url"] == "https://disk.yandex.ru/i/example"


def test_upload_document_with_display_name(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(documents.router, prefix="/documents")

    staging_dir = tmp_path / "staging"
    settings = Settings(ingest_staging_dir=str(staging_dir), ingest_sync=False)
    staging_storage = LocalStagingStorage(settings)
    captured: dict = {}

    class FakeCelery:
        def enqueue_index_document(self, path: str, *, display_name: str | None = None) -> str:
            captured["display_name"] = display_name
            return "task-123"

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        settings=settings,
        staging_storage=staging_storage,
        archive_storage=MagicMock(),
        celery_client=FakeCelery(),
        ingestion_pipeline=None,
    )

    client = TestClient(app)
    response = client.post(
        "/documents",
        files={"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")},
        data={"display_name": "Attention Is All You Need"},
    )

    assert response.status_code == 202
    assert response.json()["display_name"] == "Attention Is All You Need"
    assert captured["display_name"] == "Attention Is All You Need"


def test_upload_document_storage_error_returns_503() -> None:
    """Staging errors must return 503, not 500."""
    app = FastAPI()
    app.include_router(documents.router, prefix="/documents")

    class FailingStaging:
        def save(self, filename: str, content: bytes):
            raise RuntimeError("disk full")

    settings = Settings(ingest_sync=False)
    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        settings=settings,
        staging_storage=FailingStaging(),
        archive_storage=MagicMock(),
        celery_client=None,
        ingestion_pipeline=None,
    )

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/documents",
        files={"file": ("paper.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Document storage temporarily unavailable"


def test_download_source_file_storage_error_returns_503(tmp_path: Path) -> None:
    """Storage errors during download must return 503, not 500."""
    app = FastAPI()
    app.include_router(documents.router, prefix="/documents")

    settings = Settings(researches_dir=str(tmp_path), ingest_sync=False)
    archive_storage = FileStorage(settings)
    stored = archive_storage.save("paper.pdf", b"%PDF-1.4 content")

    class FailingArchive:
        def describe(self, filename: str):
            return archive_storage.describe(filename)

        def list_pdfs(self):
            return archive_storage.list_pdfs()

        def read(self, filename: str) -> bytes:
            raise RuntimeError("Storage backend unavailable")

        def save(self, filename: str, content: bytes):
            return archive_storage.save(filename, content)

        def delete(self, filename: str) -> None:
            archive_storage.delete(filename)

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        settings=settings,
        archive_storage=FailingArchive(),
        ingestion_pipeline=None,
    )

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get(f"/documents/files/{stored.research_id}")

    assert response.status_code == 503
    assert response.json()["detail"] == "Document storage temporarily unavailable"
