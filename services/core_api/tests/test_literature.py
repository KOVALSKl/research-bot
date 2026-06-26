from pathlib import Path

import pytest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from research_shared.config.settings import Settings
from research_shared.literature.models import ExternalPaper
from research_shared.literature.pdf_service import ExternalPdfService

from core_api.api.routes import literature
from core_api.dependencies import get_app_state


def _paper(title: str) -> ExternalPaper:
    return ExternalPaper(
        title=title,
        authors=["Alice"],
        year=2023,
        abstract="Sample abstract.",
        doi="10.1234/test",
        url="https://example.org/paper",
        source="openalex",
    )


def _client(
    *,
    settings: Settings | None = None,
    papers: list[ExternalPaper] | None = None,
    tmp_path: Path | None = None,
) -> tuple[TestClient, dict]:
    app = FastAPI()
    app.include_router(literature.router, prefix="/literature")
    captured: dict = {}
    settings = settings or Settings(_env_file=None)
    if tmp_path is not None:
        settings = settings.model_copy(update={"external_pdf_cache_dir": str(tmp_path)})

    class FakeLiteratureService:
        async def search_external(self, query: str, limit=None, year_from=None):
            captured["query"] = query
            captured["limit"] = limit
            captured["year_from"] = year_from
            return papers or [_paper("Test Paper")]

    external_pdf_service = ExternalPdfService(settings)

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        literature_service=FakeLiteratureService(),
        external_pdf_service=external_pdf_service,
        settings=settings,
    )
    return TestClient(app), captured


def test_literature_search_returns_papers() -> None:
    client, _ = _client()
    response = client.post("/literature/search", json={"query": "gnn bankruptcy"})

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["papers"][0]["title"] == "Test Paper"
    assert data["papers"][0]["source"] == "openalex"


def test_literature_search_missing_query_returns_422() -> None:
    client, _ = _client()
    response = client.post("/literature/search", json={})

    assert response.status_code == 422


def test_literature_search_default_limit_from_settings() -> None:
    settings = Settings(_env_file=None, literature_default_limit=15)
    client, captured = _client(settings=settings)

    response = client.post("/literature/search", json={"query": "test"})

    assert response.status_code == 200
    assert captured["limit"] == 15


def test_literature_search_explicit_limit() -> None:
    client, captured = _client()

    response = client.post("/literature/search", json={"query": "test", "limit": 7})

    assert response.status_code == 200
    assert captured["limit"] == 7


def test_download_external_pdf_from_cache(tmp_path: Path) -> None:
    client, _ = _client(tmp_path=tmp_path)
    cache_key = "abc123"
    settings = Settings(_env_file=None, external_pdf_cache_dir=str(tmp_path))
    service = ExternalPdfService(settings)
    service.cache.write(cache_key, b"%PDF-cached")

    app = FastAPI()
    app.include_router(literature.router, prefix="/literature")
    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        literature_service=SimpleNamespace(),
        external_pdf_service=service,
        settings=settings,
    )
    client = TestClient(app)

    response = client.get(f"/literature/papers/{cache_key}/pdf")

    assert response.status_code == 200
    assert response.content == b"%PDF-cached"
    assert response.headers["content-type"] == "application/pdf"


def test_download_external_pdf_missing_without_pdf_url(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, external_pdf_cache_dir=str(tmp_path))
    service = ExternalPdfService(settings)

    app = FastAPI()
    app.include_router(literature.router, prefix="/literature")
    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        literature_service=SimpleNamespace(),
        external_pdf_service=service,
        settings=settings,
    )
    client = TestClient(app)

    response = client.get("/literature/papers/missing-key/pdf")

    assert response.status_code == 404


def test_download_external_pdf_lazy_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, external_pdf_cache_dir=str(tmp_path))
    service = ExternalPdfService(settings)

    async def fake_get_or_fetch(cache_key: str, pdf_url: str):
        service.cache.write(cache_key, b"%PDF-lazy")
        return b"%PDF-lazy", f"{cache_key}.pdf"

    monkeypatch.setattr(service, "get_or_fetch", fake_get_or_fetch)

    app = FastAPI()
    app.include_router(literature.router, prefix="/literature")
    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        literature_service=SimpleNamespace(),
        external_pdf_service=service,
        settings=settings,
    )
    client = TestClient(app)

    response = client.get(
        "/literature/papers/lazy-key/pdf",
        params={"pdf_url": "https://example.org/paper.pdf"},
    )

    assert response.status_code == 200
    assert response.content == b"%PDF-lazy"
