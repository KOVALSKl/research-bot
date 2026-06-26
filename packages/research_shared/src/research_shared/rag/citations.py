from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from research_shared.domain.models import Citation, SourceFileRef


class CitationGroup(BaseModel):
    research_id: str
    display_name: str
    authors: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    max_score: float = 0.0
    source_url: str | None = None


def citation_filename(citation: Citation) -> str:
    if citation.filename:
        return citation.filename
    if citation.source_path:
        return Path(citation.source_path).name
    return citation.title


def citation_display_name(citation: Citation) -> str:
    if citation.display_name:
        return citation.display_name
    if citation.filename:
        return citation.filename
    if citation.source_path:
        return Path(citation.source_path).name
    return citation.title


def citation_key(citation: Citation) -> tuple[str, int | None, str]:
    return (citation.research_id, citation.page, citation_display_name(citation))


def dedupe_citations(citations: list[Citation]) -> list[Citation]:
    """Merge citations that refer to the same page of the same file.

    Keeps the citation with the highest ``score`` for each
    ``(research_id, page, display_name)`` key and fills ``filename`` from
    ``source_path`` when missing.
    """
    best: dict[tuple[str, int | None, str], Citation] = {}
    order: list[tuple[str, int | None, str]] = []

    for citation in citations:
        filename = citation_filename(citation)
        enriched = citation.model_copy(update={"filename": filename})
        key = citation_key(enriched)
        if key not in best:
            order.append(key)
            best[key] = enriched
        elif enriched.score > best[key].score:
            best[key] = enriched

    return [best[key] for key in order]


def _document_group_key(citation: Citation) -> tuple[str, str]:
    return (citation.research_id, citation_display_name(citation))


def group_citations_by_document(citations: list[Citation]) -> list[CitationGroup]:
    """Group citations by document (research_id + display_name).

    Pages are aggregated, sorted, and deduplicated. ``max_score`` is the highest
    score among chunks of the same document. Order follows first appearance.
    """
    groups: dict[tuple[str, str], CitationGroup] = {}
    order: list[tuple[str, str]] = []

    for citation in citations:
        key = _document_group_key(citation)
        page = citation.page
        if key not in groups:
            order.append(key)
            groups[key] = CitationGroup(
                research_id=citation.research_id,
                display_name=citation_display_name(citation),
                authors=list(citation.authors),
                pages=[page] if page is not None else [],
                max_score=citation.score,
                source_url=citation.source_url,
            )
            continue

        group = groups[key]
        if page is not None and page not in group.pages:
            group.pages.append(page)
        if citation.score > group.max_score:
            group.max_score = citation.score
        if not group.authors and citation.authors:
            group.authors = list(citation.authors)
        if group.source_url is None and citation.source_url:
            group.source_url = citation.source_url

    for group in groups.values():
        group.pages.sort()

    return [groups[key] for key in order]


def build_source_files(citations: list[Citation]) -> list[SourceFileRef]:
    """Build deduplicated PDF references for cited local documents (first-seen order)."""
    seen: set[str] = set()
    refs: list[SourceFileRef] = []
    for citation in citations:
        if citation.research_id in seen:
            continue
        seen.add(citation.research_id)
        refs.append(
            SourceFileRef(
                research_id=citation.research_id,
                filename=citation_filename(citation),
                display_name=citation.display_name or citation_display_name(citation),
                path=citation.source_path,
                source_url=citation.source_url,
            )
        )
    return refs
