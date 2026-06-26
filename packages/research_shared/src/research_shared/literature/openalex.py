"""OpenAlex Works API adapter."""

from __future__ import annotations

import httpx

from research_shared.literature.models import ExternalPaper
from research_shared.logging_config import get_logger

logger = get_logger(__name__)

_OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions.append((idx, word))
    if not positions:
        return ""
    positions.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positions)


def _normalize_doi(raw: str | None) -> str | None:
    if not raw:
        return None
    doi = raw.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix) :]
    return doi or None


def _extract_pdf_url(work: dict) -> str | None:
    best_oa = work.get("best_oa_location") or {}
    pdf_url = (best_oa.get("pdf_url") or "").strip()
    if pdf_url:
        return pdf_url

    open_access = work.get("open_access") or {}
    oa_url = (open_access.get("oa_url") or "").strip()
    if oa_url and (oa_url.lower().endswith(".pdf") or "/pdf/" in oa_url.lower()):
        return oa_url

    primary_location = work.get("primary_location") or {}
    primary_pdf = (primary_location.get("pdf_url") or "").strip()
    return primary_pdf or None


def _map_work(work: dict) -> ExternalPaper | None:
    title = (work.get("display_name") or work.get("title") or "").strip()
    if not title:
        return None

    authors: list[str] = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        name = (author.get("display_name") or "").strip()
        if name:
            authors.append(name)

    primary_location = work.get("primary_location") or {}
    url = (
        primary_location.get("landing_page_url")
        or work.get("id")
        or ""
    ).strip()
    if not url:
        return None

    abstract = work.get("abstract")
    if isinstance(abstract, str):
        abstract_text = abstract.strip()
    else:
        abstract_text = _reconstruct_abstract(work.get("abstract_inverted_index"))

    return ExternalPaper(
        title=title,
        authors=authors,
        year=work.get("publication_year"),
        abstract=abstract_text,
        doi=_normalize_doi(work.get("doi")),
        url=url,
        pdf_url=_extract_pdf_url(work),
        source="openalex",
    )


class OpenAlexLiteratureProvider:
    def __init__(
        self,
        *,
        base_url: str = _OPENALEX_WORKS_URL,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._client = client

    async def search(
        self,
        query: str,
        limit: int,
        year_from: int | None = None,
    ) -> list[ExternalPaper]:
        params: dict[str, str | int] = {
            "search": query,
            "per_page": limit,
        }
        if year_from is not None:
            params["filter"] = f"from_publication_date:{year_from}-01-01"

        own_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._timeout_seconds)
        try:
            response = await client.get(self._base_url, params=params)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "OpenAlex search failed",
                extra={"error": str(exc), "event": "literature.openalex.error"},
            )
            return []
        finally:
            if own_client:
                await client.aclose()

        results: list[ExternalPaper] = []
        for work in payload.get("results") or []:
            if not isinstance(work, dict):
                continue
            paper = _map_work(work)
            if paper is not None:
                results.append(paper)
        return results[:limit]
