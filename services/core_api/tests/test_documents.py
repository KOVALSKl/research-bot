from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from research_shared.config.settings import Settings
from research_shared.domain.models import DocumentRecord, IngestStatus
from research_shared.ingestion.file_storage import FileStorage

from core_api.api.routes import documents
from core_api.dependencies import get_app_state


def test_upload_document_returns_202(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(documents.router, prefix="/documents")

    settings = Settings(researches_dir=str(tmp_path), ingest_sync=False)
    file_storage = FileStorage(settings)

    class FakeCelery:
        def enqueue_index_document(self, path: str, *, display_name: str | None = None) -> str:
            return "task-123"

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        settings=settings,
        file_storage=file_storage,
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
    assert (tmp_path / "paper.pdf").exists()


def test_download_source_file_by_research_id(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(documents.router, prefix="/documents")

    settings = Settings(researches_dir=str(tmp_path), ingest_sync=False)
    file_storage = FileStorage(settings)
    stored = file_storage.save("paper.pdf", b"%PDF-1.4 content")

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        settings=settings,
        file_storage=file_storage,
        ingestion_pipeline=None,
    )

    client = TestClient(app)
    response = client.get(f"/documents/files/{stored.research_id}")

    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 content"
    assert response.headers["content-type"] == "application/pdf"


def test_download_source_file_unknown_research_id(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(documents.router, prefix="/documents")

    settings = Settings(researches_dir=str(tmp_path), ingest_sync=False)
    file_storage = FileStorage(settings)

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        settings=settings,
        file_storage=file_storage,
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
        file_storage=FileStorage(settings),
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
        file_storage=FileStorage(settings),
        ingestion_pipeline=FakePipeline(),
    )

    client = TestClient(app)
    response = client.get("/documents", params={"status": "indexed"})

    assert response.status_code == 200
    data = response.json()["documents"]
    assert len(data) == 1
    assert data[0]["display_name"] == "My Paper"
    assert data[0]["chunk_count"] == 10


def test_upload_document_with_display_name(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(documents.router, prefix="/documents")

    settings = Settings(researches_dir=str(tmp_path), ingest_sync=False)
    file_storage = FileStorage(settings)
    captured: dict = {}

    class FakeCelery:
        def enqueue_index_document(self, path: str, *, display_name: str | None = None) -> str:
            captured["display_name"] = display_name
            return "task-123"

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        settings=settings,
        file_storage=file_storage,
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
