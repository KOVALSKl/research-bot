"""Semantic Scholar Graph API adapter (optional, requires API key)."""

from __future__ import annotations

import httpx

from research_shared.config.settings import Settings
from research_shared.literature.models import ExternalPaper
from research_shared.logging_config import get_logger

logger = get_logger(__name__)

_SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def _normalize_doi(raw: str | None) -> str | None:
    if not raw:
        return None
    doi = raw.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix) :]
    return doi or None


def _map_paper(item: dict) -> ExternalPaper | None:
    title = (item.get("title") or "").strip()
    if not title:
        return None

    authors: list[str] = []
    for author in item.get("authors") or []:
        name = (author.get("name") or "").strip()
        if name:
            authors.append(name)

    external_ids = item.get("externalIds") or {}
    doi = _normalize_doi(external_ids.get("DOI"))
    paper_id = item.get("paperId") or ""
    url = (item.get("url") or "").strip()
    if not url and paper_id:
        url = f"https://www.semanticscholar.org/paper/{paper_id}"
    if not url:
        return None

    open_access_pdf = item.get("openAccessPdf") or {}
    pdf_url = None
    if isinstance(open_access_pdf, dict):
        pdf_url = (open_access_pdf.get("url") or "").strip() or None

    return ExternalPaper(
        title=title,
        authors=authors,
        year=item.get("year"),
        abstract=(item.get("abstract") or "").strip(),
        doi=doi,
        url=url,
        pdf_url=pdf_url,
        source="semantic_scholar",
    )


class SemanticScholarLiteratureProvider:
    def __init__(
        self,
        settings: Settings,
        *,
        base_url: str = _SEMANTIC_SCHOLAR_SEARCH_URL,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._client = client

    async def search(
        self,
        query: str,
        limit: int,
        year_from: int | None = None,
    ) -> list[ExternalPaper]:
        api_key = self._settings.semantic_scholar_api_key.strip()
        if not api_key:
            return []

        params: dict[str, str | int] = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,abstract,externalIds,url,paperId,openAccessPdf",
        }
        if year_from is not None:
            params["year"] = f"{year_from}-"

        headers = {"x-api-key": api_key}

        own_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._timeout_seconds)
        try:
            response = await client.get(self._base_url, params=params, headers=headers)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            logger.warning(
                "Semantic Scholar search failed",
                extra={"error": str(exc), "event": "literature.semantic_scholar.error"},
            )
            return []
        except ValueError as exc:
            logger.warning(
                "Semantic Scholar response parse failed",
                extra={"error": str(exc), "event": "literature.semantic_scholar.error"},
            )
            return []
        finally:
            if own_client:
                await client.aclose()

        papers: list[ExternalPaper] = []
        for item in payload.get("data") or []:
            if not isinstance(item, dict):
                continue
            paper = _map_paper(item)
            if paper is not None:
                papers.append(paper)
        return papers[:limit]
