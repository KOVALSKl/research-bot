from __future__ import annotations

import httpx

from research_shared.config.settings import Settings
from research_shared.ingestion.factory import create_archive_storage, create_document_storage
from research_shared.ingestion.yandex_disk import YandexDiskStorage
from research_shared.literature.storage.yandex_disk_cache import YandexDiskPdfCache


def _yandex_handler(uploaded: dict[str, bytes], resources: dict[str, dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if (
            "/v1/disk/resources" in url
            and request.method == "PUT"
            and "upload" not in url
            and "publish" not in url
        ):
            return httpx.Response(201)
        if "/resources/upload" in url and request.method == "GET":
            path = request.url.params.get("path", "")
            return httpx.Response(200, json={"href": f"https://upload.example/put?path={path}"})
        if url.startswith("https://upload.example/put") and request.method == "PUT":
            path = request.url.params.get("path", "")
            uploaded[path] = request.content
            name = path.rsplit("/", 1)[-1]
            resources[path] = {
                "type": "file",
                "name": name,
                "size": len(request.content),
                "md5": "abc123",
                "modified": "2026-06-21T10:00:00Z",
            }
            return httpx.Response(201)
        if "/resources/download" in url and request.method == "GET":
            path = request.url.params.get("path", "")
            return httpx.Response(200, json={"href": f"https://download.example/get?path={path}"})
        if "/resources/publish" in url and request.method == "PUT":
            path = request.url.params.get("path", "")
            if path in resources:
                resources[path] = {**resources[path], "public_url": f"https://disk.yandex.ru/i/{path.rsplit('/', 1)[-1]}"}
            return httpx.Response(200)
        if url.startswith("https://download.example/get") and request.method == "GET":
            path = request.url.params.get("path", "")
            return httpx.Response(200, content=uploaded.get(path, b""))
        if "/v1/disk/resources" in url and request.method == "GET" and "upload" not in url and "download" not in url:
            path = request.url.params.get("path", "")
            if path.endswith(".pdf"):
                item = resources.get(path)
                if item:
                    return httpx.Response(200, json=item)
                return httpx.Response(404)
            items = [
                item
                for key, item in resources.items()
                if key.startswith(path.rstrip("/") + "/") or key == path
            ]
            return httpx.Response(200, json={"_embedded": {"items": items}})
        return httpx.Response(404)

    return handler


def test_create_document_storage_local_by_default() -> None:
    settings = Settings(_env_file=None, storage_backend="local")
    storage = create_document_storage(settings)
    assert storage.__class__.__name__ == "LocalDocumentStorage"


def test_create_document_storage_yandex_with_token() -> None:
    settings = Settings(
        _env_file=None,
        storage_backend="yandex",
        yandex_disk_api_token="test-token",
    )
    storage = create_archive_storage(settings)
    assert isinstance(storage, YandexDiskStorage)
    assert isinstance(create_document_storage(settings), YandexDiskStorage)


def test_create_document_storage_falls_back_without_token() -> None:
    settings = Settings(_env_file=None, storage_backend="yandex", yandex_disk_api_token="")
    storage = create_document_storage(settings)
    assert storage.__class__.__name__ == "LocalDocumentStorage"


def test_yandex_disk_save_list_read_roundtrip() -> None:
    uploaded: dict[str, bytes] = {}
    resources: dict[str, dict] = {}
    transport = httpx.MockTransport(_yandex_handler(uploaded, resources))
    storage = YandexDiskStorage("token", "disk:/research-docs", transport=transport)

    stored = storage.save("paper.pdf", b"%PDF-1.4 test")
    assert stored.filename == "paper.pdf"
    assert stored.research_id

    listed = storage.list_pdfs()
    assert len(listed) == 1
    assert listed[0].filename == "paper.pdf"

    content = storage.read("paper.pdf")
    assert content == b"%PDF-1.4 test"


def test_yandex_disk_publish_and_public_url() -> None:
    uploaded: dict[str, bytes] = {}
    resources: dict[str, dict] = {}
    transport = httpx.MockTransport(_yandex_handler(uploaded, resources))
    storage = YandexDiskStorage("token", "disk:/research-docs", transport=transport)

    storage.save("paper.pdf", b"%PDF-1.4 test")
    storage.publish("paper.pdf")
    public_url = storage.get_public_url("paper.pdf")

    assert public_url == "https://disk.yandex.ru/i/paper.pdf"


def test_yandex_disk_pdf_cache_write_read() -> None:
    uploaded: dict[str, bytes] = {}
    resources: dict[str, dict] = {}
    transport = httpx.MockTransport(_yandex_handler(uploaded, resources))
    disk = YandexDiskStorage("token", "disk:/research-docs", transport=transport)
    cache = YandexDiskPdfCache(disk)

    cache.write("key123", b"%PDF cached")
    assert cache.exists("key123")
    assert cache.read("key123") == b"%PDF cached"
