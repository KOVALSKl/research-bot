from __future__ import annotations

import json
from dataclasses import dataclass

from research_shared.literature.models import ExternalPaper
from research_shared.literature.service import (
    ExternalLiteratureSearchOutcome,
    ExternalLiteratureService,
    ProviderSearchStat,
    _dedupe_papers,
)


@dataclass(frozen=True)
class ExternalSearchResult:
    papers: list[ExternalPaper]
    context: str
    queries: tuple[str, ...] = ()
    provider_stats: tuple[ProviderSearchStat, ...] = ()
    cache_hit: bool = False
    fallback_used: bool = False


def _build_external_context(papers: list[ExternalPaper]) -> str:
    parts: list[str] = []
    for index, paper in enumerate(papers, start=1):
        year = f" ({paper.year})" if paper.year is not None else ""
        authors = ", ".join(paper.authors) if paper.authors else "—"
        abstract = paper.abstract or "—"
        parts.append(
            f"[E{index}] {paper.title}{year}\n"
            f"Authors: {authors}\n"
            f"Abstract: {abstract}"
        )
    return "\n\n".join(parts)


def _format_provider_stats(stats: tuple[ProviderSearchStat, ...]) -> str:
    if not stats:
        return ""
    parts: list[str] = []
    for stat in stats:
        if stat.error:
            parts.append(f"{stat.provider}:0({stat.error})")
        else:
            parts.append(f"{stat.provider}:{stat.count}")
    return ";".join(parts)


def format_external_search_detail(result: ExternalSearchResult) -> str:
    payload = {
        "queries": list(result.queries),
        "providers": _format_provider_stats(result.provider_stats),
        "cache_hit": result.cache_hit,
        "fallback_used": result.fallback_used,
        "count": len(result.papers),
    }
    return json.dumps(payload, ensure_ascii=False)


def _merge_outcomes(*outcomes: ExternalLiteratureSearchOutcome) -> ExternalLiteratureSearchOutcome:
    merged_papers: list[ExternalPaper] = []
    stats: list[ProviderSearchStat] = []
    queries: list[str] = []
    cache_hit = True
    for outcome in outcomes:
        merged_papers.extend(outcome.papers)
        stats.extend(outcome.provider_stats)
        queries.extend(outcome.queries)
        cache_hit = cache_hit and outcome.cache_hit

    unique_queries = tuple(dict.fromkeys(queries))
    return ExternalLiteratureSearchOutcome(
        papers=_dedupe_papers(merged_papers),
        provider_stats=tuple(stats),
        cache_hit=cache_hit,
        queries=unique_queries,
    )


async def external_literature_search(
    literature_service: ExternalLiteratureService,
    query: str,
    limit: int,
) -> ExternalSearchResult:
    return await external_literature_search_queries(
        literature_service,
        [query],
        limit,
        fallback_query=None,
    )


async def external_literature_search_queries(
    literature_service: ExternalLiteratureService,
    queries: list[str],
    limit: int,
    *,
    fallback_query: str | None = None,
    mode: str | None = None,
) -> ExternalSearchResult:
    unique_queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
    outcome = await literature_service.search_external_queries(
        unique_queries,
        limit=limit,
        mode=mode,
    )

    fallback_used = False
    if not outcome.papers and fallback_query:
        fallback = fallback_query.strip()
        if fallback and fallback.casefold() not in {q.casefold() for q in unique_queries}:
            fallback_outcome = await literature_service.search_with_diagnostics(
                fallback,
                limit=limit,
                mode=mode,
            )
            outcome = _merge_outcomes(outcome, fallback_outcome)
            fallback_used = True

    papers = outcome.papers[:limit]
    return ExternalSearchResult(
        papers=papers,
        context=_build_external_context(papers),
        queries=outcome.queries,
        provider_stats=outcome.provider_stats,
        cache_hit=outcome.cache_hit,
        fallback_used=fallback_used,
    )
