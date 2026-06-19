from __future__ import annotations

from pathlib import Path

from research_shared.domain.models import Citation


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
