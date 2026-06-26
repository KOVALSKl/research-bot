"""arXiv Atom feed adapter."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime

import httpx

from research_shared.literature.models import ExternalPaper
from research_shared.logging_config import get_logger

logger = get_logger(__name__)

_ARXIV_API_URL = "http://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _arxiv_pdf_url(url: str) -> str | None:
    match = re.search(r"arxiv\.org/abs/([\d.]+(?:v\d+)?)", url, re.IGNORECASE)
    if match:
        return f"https://arxiv.org/pdf/{match.group(1)}.pdf"
    match = re.search(r"arxiv\.org/pdf/([\d.]+(?:v\d+)?)", url, re.IGNORECASE)
    if match:
        return f"https://arxiv.org/pdf/{match.group(1)}.pdf"
    return None


def _parse_year(published: str | None) -> int | None:
    if not published:
        return None
    try:
        return datetime.fromisoformat(published.replace("Z", "+00:00")).year
    except ValueError:
        match = re.match(r"(\d{4})", published)
        return int(match.group(1)) if match else None


def _entry_to_paper(entry: ET.Element) -> ExternalPaper | None:
    title_el = entry.find("atom:title", _ATOM_NS)
    title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
    if not title:
        return None

    authors: list[str] = []
    for author_el in entry.findall("atom:author", _ATOM_NS):
        name_el = author_el.find("atom:name", _ATOM_NS)
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())

    published_el = entry.find("atom:published", _ATOM_NS)
    published = published_el.text.strip() if published_el is not None and published_el.text else None
    year = _parse_year(published)

    summary_el = entry.find("atom:summary", _ATOM_NS)
    abstract = ""
    if summary_el is not None and summary_el.text:
        abstract = summary_el.text.strip().replace("\n", " ")

    id_el = entry.find("atom:id", _ATOM_NS)
    url = (id_el.text or "").strip() if id_el is not None else ""
    if not url:
        link_el = entry.find("atom:link[@rel='alternate']", _ATOM_NS)
        if link_el is not None:
            url = (link_el.get("href") or "").strip()
    if not url:
        return None

    doi_el = entry.find("arxiv:doi", _ATOM_NS)
    doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

    return ExternalPaper(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        doi=doi,
        url=url,
        pdf_url=_arxiv_pdf_url(url),
        source="arxiv",
    )


def _parse_atom_feed(xml_text: str) -> list[ExternalPaper]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    papers: list[ExternalPaper] = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        paper = _entry_to_paper(entry)
        if paper is not None:
            papers.append(paper)
    return papers


def _is_likely_cyrillic(query: str) -> bool:
    """Return True when the query is predominantly Cyrillic (arXiv doesn't support Russian queries)."""
    chars = [c for c in query if c.isalpha()]
    if not chars:
        return False
    cyrillic = sum(1 for c in chars if "Ѐ" <= c <= "ӿ")
    return cyrillic / len(chars) > 0.5


class ArxivLiteratureProvider:
    def __init__(
        self,
        *,
        base_url: str = _ARXIV_API_URL,
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
        if _is_likely_cyrillic(query):
            logger.debug(
                "arXiv search skipped: Cyrillic query",
                extra={"event": "literature.arxiv.skip_cyrillic", "query": query},
            )
            return []

        params = {
            "search_query": f"all:{query}",
            "max_results": limit,
        }

        own_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._timeout_seconds)
        try:
            response = await client.get(self._base_url, params=params)
            response.raise_for_status()
            papers = _parse_atom_feed(response.text)
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "arXiv search failed",
                extra={"error": str(exc), "event": "literature.arxiv.error"},
            )
            return []
        finally:
            if own_client:
                await client.aclose()

        if year_from is not None:
            papers = [p for p in papers if p.year is None or p.year >= year_from]
        return papers[:limit]
