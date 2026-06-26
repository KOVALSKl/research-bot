from __future__ import annotations

import httpx

from research_shared.config.settings import Settings
from research_shared.http.redirects import fetch_with_redirects
from research_shared.literature.storage.protocols import PdfCacheStorage
from research_shared.literature.storage.yandex_disk_cache import create_pdf_cache_storage
from research_shared.logging_config import get_logger

logger = get_logger(__name__)

_PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",
}


class ExternalPdfService:
    def __init__(
        self,
        settings: Settings,
        cache: PdfCacheStorage | None = None,
    ) -> None:
        self._settings = settings
        self._cache = cache or create_pdf_cache_storage(settings)

    @property
    def cache(self) -> PdfCacheStorage:
        return self._cache

    async def get_or_fetch(
        self,
        cache_key: str,
        pdf_url: str,
    ) -> tuple[bytes, str] | None:
        if not self._settings.external_pdf_fetch_enabled:
            return None

        cached = self._cache.read(cache_key)
        if cached is not None:
            return cached, f"{cache_key}.pdf"

        try:
            content = await self._fetch_pdf(pdf_url)
        except Exception:
            logger.exception(
                "External PDF fetch failed",
                extra={
                    "cache_key": cache_key,
                    "event": "external_pdf.fetch_failed",
                },
            )
            return None

        if len(content) > self._settings.external_pdf_max_bytes:
            logger.warning(
                "External PDF exceeds size limit",
                extra={
                    "cache_key": cache_key,
                    "count": len(content),
                    "event": "external_pdf.skip_oversized",
                },
            )
            return None

        self._cache.write(cache_key, content)
        return content, f"{cache_key}.pdf"

    async def _fetch_pdf(self, url: str) -> bytes:
        headers = {"User-Agent": "research-bot/1.0"}
        async with httpx.AsyncClient(
            timeout=self._settings.external_pdf_fetch_timeout_seconds,
            follow_redirects=True,
            max_redirects=self._settings.external_pdf_max_redirects,
            headers=headers,
        ) as client:
            try:
                response = await client.get(url)
            except httpx.TooManyRedirects:
                response = await fetch_with_redirects(
                    client,
                    url,
                    max_redirects=self._settings.external_pdf_max_redirects,
                    headers=headers,
                )

            if response.status_code in {301, 302, 303, 307, 308}:
                response = await fetch_with_redirects(
                    client,
                    url,
                    max_redirects=self._settings.external_pdf_max_redirects,
                    headers=headers,
                )

            response.raise_for_status()
            content_type = (response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
            if content_type and content_type not in _PDF_CONTENT_TYPES:
                logger.warning(
                    "Unexpected external PDF content type",
                    extra={
                        "content_type": content_type,
                        "event": "external_pdf.unexpected_content_type",
                    },
                )
            return response.content
