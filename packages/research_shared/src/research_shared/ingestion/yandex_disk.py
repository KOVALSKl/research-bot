from __future__ import annotations

import hashlib
import time
from typing import Any

import httpx

from research_shared.ingestion.storage_protocol import PdfFileInfo, StoredFile
from research_shared.logging_config import get_logger

logger = get_logger(__name__)

_API_BASE = "https://cloud-api.yandex.net/v1/disk"
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3


def compute_content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def compute_research_id(content: bytes) -> str:
    return compute_content_hash(content)[:16]


class YandexDiskStorage:
    """Document storage backed by Yandex Disk REST API."""

    def __init__(
        self,
        token: str,
        base_path: str = "disk:/research-docs",
        *,
        timeout_seconds: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not token.strip():
            raise ValueError("YANDEX_DISK_API_TOKEN is required for Yandex Disk storage")
        self._token = token.strip()
        self._base_path = base_path.rstrip("/")
        self._timeout = timeout_seconds
        self._transport = transport

    @property
    def base_path(self) -> str:
        return self._base_path

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"OAuth {self._token}"}

    def _resource_path(self, filename: str) -> str:
        safe_name = filename.split("/")[-1]
        return f"{self._base_path}/{safe_name}"

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        content: bytes | None = None,
        follow_redirects: bool = True,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                client_kwargs: dict[str, Any] = {
                    "timeout": self._timeout,
                    "follow_redirects": follow_redirects,
                }
                if self._transport is not None:
                    client_kwargs["transport"] = self._transport
                with httpx.Client(**client_kwargs) as client:
                    response = client.request(
                        method,
                        url,
                        params=params,
                        headers=self._headers(),
                        content=content,
                    )
                if response.status_code in _RETRYABLE_STATUSES and attempt < _MAX_RETRIES - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("Yandex Disk request failed without response")

    def ensure_folder(self, path: str | None = None) -> None:
        target = path or self._base_path
        response = self._request(
            "PUT",
            f"{_API_BASE}/resources",
            params={"path": target},
        )
        if response.status_code in {201, 409}:
            return
        response.raise_for_status()

    def save(self, filename: str, content: bytes) -> StoredFile:
        self.ensure_folder()
        safe_name = filename.split("/")[-1]
        resource_path = self._resource_path(safe_name)

        upload_response = self._request(
            "GET",
            f"{_API_BASE}/resources/upload",
            params={"path": resource_path, "overwrite": "true"},
            follow_redirects=False,
        )
        upload_response.raise_for_status()
        href = upload_response.json().get("href")
        if not href:
            raise RuntimeError(f"Yandex Disk upload href missing for {resource_path}")

        put_response = self._request("PUT", href, content=content, follow_redirects=True)
        put_response.raise_for_status()

        content_hash = compute_content_hash(content)
        return StoredFile(
            filename=safe_name,
            content_hash=content_hash,
            research_id=compute_research_id(content),
            path=None,
        )

    def list_pdfs(self) -> list[PdfFileInfo]:
        self.ensure_folder()
        response = self._request(
            "GET",
            f"{_API_BASE}/resources",
            params={"path": self._base_path, "limit": 1000},
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        payload = response.json()
        items: list[PdfFileInfo] = []
        for item in payload.get("_embedded", {}).get("items", []):
            if item.get("type") != "file":
                continue
            name = str(item.get("name", ""))
            if not name.lower().endswith(".pdf"):
                continue
            items.append(
                PdfFileInfo(
                    filename=name,
                    size=int(item.get("size") or 0),
                    modified=item.get("modified"),
                    content_hash=str(item.get("md5") or "") or None,
                )
            )
        return sorted(items, key=lambda info: info.filename.lower())

    def read(self, filename: str) -> bytes:
        resource_path = self._resource_path(filename)
        download_response = self._request(
            "GET",
            f"{_API_BASE}/resources/download",
            params={"path": resource_path},
            follow_redirects=False,
        )
        if download_response.status_code == 404:
            raise FileNotFoundError(filename)
        download_response.raise_for_status()
        href = download_response.json().get("href")
        if not href:
            raise RuntimeError(f"Yandex Disk download href missing for {resource_path}")

        content_response = self._request("GET", href, follow_redirects=True)
        content_response.raise_for_status()
        return content_response.content

    def describe(self, filename: str) -> StoredFile:
        resource_path = self._resource_path(filename)
        response = self._request(
            "GET",
            f"{_API_BASE}/resources",
            params={"path": resource_path},
        )
        if response.status_code == 404:
            raise FileNotFoundError(filename)
        response.raise_for_status()
        item = response.json()
        content_hash = str(item.get("md5") or "")
        if not content_hash:
            content = self.read(filename)
            content_hash = compute_content_hash(content)
            research_id = compute_research_id(content)
        else:
            research_id = content_hash[:16]
        return StoredFile(
            filename=filename.split("/")[-1],
            content_hash=content_hash,
            research_id=research_id,
            path=None,
        )

    def delete(self, filename: str) -> None:
        resource_path = self._resource_path(filename)
        response = self._request(
            "DELETE",
            f"{_API_BASE}/resources",
            params={"path": resource_path, "permanently": "true"},
        )
        if response.status_code in {204, 404}:
            return
        response.raise_for_status()

    def publish(self, filename: str) -> None:
        resource_path = self._resource_path(filename)
        response = self._request(
            "PUT",
            f"{_API_BASE}/resources/publish",
            params={"path": resource_path},
        )
        # 409 means the file is already published — treat as success
        if response.status_code in {200, 204, 409}:
            return
        response.raise_for_status()

    def get_public_url(self, filename: str) -> str | None:
        resource_path = self._resource_path(filename)
        response = self._request(
            "GET",
            f"{_API_BASE}/resources",
            params={"path": resource_path, "fields": "public_url"},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        public_url = response.json().get("public_url")
        return str(public_url) if public_url else None

    def publish_and_get_url(self, filename: str) -> str | None:
        """Publish the file and return its public Yandex Disk URL in one operation.

        Uses the href from the publish response to avoid rebuilding the resource
        path for the follow-up metadata query.
        """
        resource_path = self._resource_path(filename)
        publish_response = self._request(
            "PUT",
            f"{_API_BASE}/resources/publish",
            params={"path": resource_path},
        )
        # 409 = already published; still proceed to fetch the URL
        if publish_response.status_code not in {200, 204, 409}:
            publish_response.raise_for_status()

        # Use href from response to fetch public_url, falling back to direct path query
        resource_href = publish_response.json().get("href") if publish_response.content else None
        if resource_href:
            separator = "&" if "?" in resource_href else "?"
            meta_url = f"{resource_href}{separator}fields=public_url"
            meta_response = self._request("GET", meta_url)
        else:
            meta_response = self._request(
                "GET",
                f"{_API_BASE}/resources",
                params={"path": resource_path, "fields": "public_url"},
            )

        if meta_response.status_code == 404:
            return None
        meta_response.raise_for_status()
        public_url = meta_response.json().get("public_url")
        return str(public_url) if public_url else None
