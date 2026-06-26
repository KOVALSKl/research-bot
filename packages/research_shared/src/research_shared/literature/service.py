"""External literature aggregation: parallel search, dedupe, Redis cache."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import redis.asyncio as aioredis

from research_shared.config.settings import Settings
from research_shared.literature.arxiv import ArxivLiteratureProvider
from research_shared.literature.models import ExternalPaper
from research_shared.literature.openalex import OpenAlexLiteratureProvider
from research_shared.literature.protocols import LiteratureSearchProvider
from research_shared.literature.ranking import (
    post_filter_external_papers,
    rerank_external_papers,
    rerank_external_papers_multi_query,
    score_external_paper_multi_query,
)
from research_shared.literature.semantic_scholar import SemanticScholarLiteratureProvider
from research_shared.logging_config import get_logger

logger = get_logger(__name__)

_CACHE_PREFIX = "literature:"


@dataclass(frozen=True)
class ProviderSearchStat:
    provider: str
    count: int
    error: str | None = None


@dataclass(frozen=True)
class ExternalLiteratureSearchOutcome:
    papers: list[ExternalPaper]
    provider_stats: tuple[ProviderSearchStat, ...] = ()
    cache_hit: bool = False
    queries: tuple[str, ...] = ()


def _normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    normalized = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized or None


def _normalize_title(title: str) -> str:
    lowered = title.lower().strip()
    return re.sub(r"[\W_]+", "", lowered, flags=re.UNICODE)


def _cache_key(query: str, limit: int, year_from: int | None) -> str:
    raw = f"{query}|{limit}|{year_from}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{_CACHE_PREFIX}{digest}"


def _cache_key_queries(queries: list[str], limit: int, year_from: int | None) -> str:
    normalized = "|".join(sorted(q.strip() for q in queries if q.strip()))
    raw = f"multi:{normalized}|{limit}|{year_from}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{_CACHE_PREFIX}{digest}"


def _dedupe_papers(papers: list[ExternalPaper]) -> list[ExternalPaper]:
    seen_doi: set[str] = set()
    seen_title: set[str] = set()
    unique: list[ExternalPaper] = []

    for paper in papers:
        doi_key = _normalize_doi(paper.doi)
        if doi_key:
            if doi_key in seen_doi:
                continue
            seen_doi.add(doi_key)
            unique.append(paper)
            continue

        title_key = _normalize_title(paper.title)
        if title_key and title_key in seen_title:
            continue
        if title_key:
            seen_title.add(title_key)
        unique.append(paper)

    return unique


def _finalize_external_papers(
    papers: list[ExternalPaper],
    queries: list[str],
    limit: int,
    *,
    min_score: float = 0.15,
) -> list[ExternalPaper]:
    primary_query = queries[0] if queries else ""
    deduped = _dedupe_papers(papers)
    if len(queries) > 1:
        ranked = rerank_external_papers_multi_query(deduped, queries)
    else:
        ranked = rerank_external_papers(deduped, primary_query)
    filtered = post_filter_external_papers(ranked, primary_query, min_score=min_score)
    return filtered[:limit]


def create_literature_service(
    settings: Settings,
    redis_client: aioredis.Redis | None = None,
    providers: list[LiteratureSearchProvider] | None = None,
) -> ExternalLiteratureService:
    if providers is None:
        providers = [
            OpenAlexLiteratureProvider(),
            ArxivLiteratureProvider(),
            SemanticScholarLiteratureProvider(settings),
        ]
    return ExternalLiteratureService(settings, providers, redis_client=redis_client)


class ExternalLiteratureService:
    def __init__(
        self,
        settings: Settings,
        providers: list[LiteratureSearchProvider],
        *,
        redis_client: aioredis.Redis | None = None,
    ) -> None:
        self._settings = settings
        self._providers = providers
        self._redis = redis_client

    async def search_external(
        self,
        query: str,
        limit: int | None = None,
        year_from: int | None = None,
    ) -> list[ExternalPaper]:
        outcome = await self.search_with_diagnostics(query, limit=limit, year_from=year_from)
        return outcome.papers

    async def search_external_queries(
        self,
        queries: list[str],
        limit: int | None = None,
        year_from: int | None = None,
        *,
        mode: str | None = None,
    ) -> ExternalLiteratureSearchOutcome:
        unique_queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
        if not unique_queries:
            return ExternalLiteratureSearchOutcome(papers=[], queries=())

        effective_limit = limit if limit is not None else self._settings.literature_default_limit
        post_filter_min_score = 0.15
        if mode == "idea_evaluation":
            effective_limit = max(
                effective_limit,
                self._settings.literature_idea_mode_limit,
            )
            post_filter_min_score = self._settings.literature_idea_post_filter_min_score

        if len(unique_queries) == 1:
            outcome = await self.search_with_diagnostics(
                unique_queries[0],
                limit=effective_limit,
                year_from=year_from,
                mode=mode,
            )
            return ExternalLiteratureSearchOutcome(
                papers=outcome.papers,
                provider_stats=outcome.provider_stats,
                cache_hit=outcome.cache_hit,
                queries=(unique_queries[0],),
            )

        cache_key = _cache_key_queries(unique_queries, effective_limit, year_from)
        cached, cache_hit = await self._read_cache(cache_key)
        if cache_hit:
            logger.info(
                "External literature cache hit (multi-query)",
                extra={
                    "cache_hit": True,
                    "count": len(cached),
                    "event": "literature.search",
                    "queries": unique_queries,
                },
            )
            return ExternalLiteratureSearchOutcome(
                papers=cached,
                cache_hit=True,
                queries=tuple(unique_queries),
            )

        provider_stats: list[ProviderSearchStat] = []
        merged: list[ExternalPaper] = []
        per_query_limit = max(2, (effective_limit + len(unique_queries) - 1) // len(unique_queries))
        for query in unique_queries:
            batch_outcome = await self._search_providers(
                query,
                per_query_limit,
                year_from,
                post_filter_min_score=post_filter_min_score,
            )
            merged.extend(batch_outcome.papers[:per_query_limit])
            provider_stats.extend(batch_outcome.provider_stats)

        deduped = _finalize_external_papers(
            merged,
            unique_queries,
            effective_limit,
            min_score=post_filter_min_score,
        )
        await self._write_cache(cache_key, deduped)

        aggregated_stats = _aggregate_provider_stats(provider_stats)
        logger.info(
            "External literature multi-query search completed",
            extra={
                "cache_hit": False,
                "count": len(deduped),
                "event": "literature.search",
                "queries": unique_queries,
            },
        )
        return ExternalLiteratureSearchOutcome(
            papers=deduped,
            provider_stats=tuple(aggregated_stats),
            cache_hit=False,
            queries=tuple(unique_queries),
        )

    async def search_with_diagnostics(
        self,
        query: str,
        limit: int | None = None,
        year_from: int | None = None,
        *,
        mode: str | None = None,
    ) -> ExternalLiteratureSearchOutcome:
        effective_limit = limit if limit is not None else self._settings.literature_default_limit
        post_filter_min_score = 0.15
        if mode == "idea_evaluation":
            effective_limit = max(
                effective_limit,
                self._settings.literature_idea_mode_limit,
            )
            post_filter_min_score = self._settings.literature_idea_post_filter_min_score
        cache_key = _cache_key(query, effective_limit, year_from)

        cached, cache_hit = await self._read_cache(cache_key)
        if cache_hit:
            logger.info(
                "External literature cache hit",
                extra={
                    "cache_hit": True,
                    "count": len(cached),
                    "event": "literature.search",
                    "query": query,
                },
            )
            return ExternalLiteratureSearchOutcome(
                papers=cached,
                cache_hit=True,
                queries=(query,),
            )

        outcome = await self._search_providers(
            query,
            effective_limit,
            year_from,
            post_filter_min_score=post_filter_min_score,
        )
        await self._write_cache(cache_key, outcome.papers)

        logger.info(
            "External literature search completed",
            extra={
                "cache_hit": False,
                "count": len(outcome.papers),
                "event": "literature.search",
                "query": query,
            },
        )
        return ExternalLiteratureSearchOutcome(
            papers=outcome.papers,
            provider_stats=outcome.provider_stats,
            cache_hit=False,
            queries=(query,),
        )

    async def _search_providers(
        self,
        query: str,
        limit: int,
        year_from: int | None,
        *,
        post_filter_min_score: float = 0.15,
    ) -> ExternalLiteratureSearchOutcome:
        provider_results = await asyncio.gather(
            *[
                self._safe_search_with_stats(provider, query, limit, year_from)
                for provider in self._providers
            ],
            return_exceptions=False,
        )

        merged: list[ExternalPaper] = []
        stats: list[ProviderSearchStat] = []
        for batch in provider_results:
            merged.extend(batch.papers)
            stats.append(batch.stat)

        deduped = _finalize_external_papers(
            merged,
            [query],
            limit,
            min_score=post_filter_min_score,
        )
        return ExternalLiteratureSearchOutcome(
            papers=deduped,
            provider_stats=tuple(stats),
            queries=(query,),
        )

    async def aclose(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()

    async def _safe_search_with_stats(
        self,
        provider: LiteratureSearchProvider,
        query: str,
        limit: int,
        year_from: int | None,
    ) -> _ProviderBatch:
        provider_name = type(provider).__name__
        try:
            papers = await provider.search(query, limit, year_from)
            logger.debug(
                "Literature provider result",
                extra={
                    "event": "literature.provider.result",
                    "provider": provider_name,
                    "query": query,
                    "raw_count": len(papers),
                },
            )
            return _ProviderBatch(
                papers=papers,
                stat=ProviderSearchStat(provider=provider_name, count=len(papers)),
            )
        except Exception as exc:
            logger.warning(
                "Literature provider failed",
                extra={
                    "error": str(exc),
                    "event": "literature.provider.error",
                    "provider": provider_name,
                },
            )
            return _ProviderBatch(
                papers=[],
                stat=ProviderSearchStat(
                    provider=provider_name,
                    count=0,
                    error=str(exc),
                ),
            )

    async def _safe_search(
        self,
        provider: LiteratureSearchProvider,
        query: str,
        limit: int,
        year_from: int | None,
    ) -> list[ExternalPaper]:
        batch = await self._safe_search_with_stats(provider, query, limit, year_from)
        return batch.papers

    async def _read_cache(self, key: str) -> tuple[list[ExternalPaper], bool]:
        if self._redis is None:
            return [], False
        try:
            raw = await self._redis.get(key)
        except Exception as exc:
            logger.warning(
                "Literature cache read failed",
                extra={"error": str(exc), "event": "literature.cache.error"},
            )
            return [], False
        if not raw:
            return [], False
        try:
            payload: list[dict[str, Any]] = json.loads(raw)
            return [ExternalPaper.model_validate(item) for item in payload], True
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Literature cache deserialize failed",
                extra={"error": str(exc), "event": "literature.cache.error"},
            )
            return [], False

    async def _write_cache(self, key: str, papers: list[ExternalPaper]) -> None:
        if self._redis is None:
            return
        try:
            payload = json.dumps([paper.model_dump(mode="json") for paper in papers])
            ttl = (
                self._settings.literature_cache_empty_ttl_seconds
                if not papers
                else self._settings.literature_cache_ttl_seconds
            )
            await self._redis.setex(key, ttl, payload)
        except Exception as exc:
            logger.warning(
                "Literature cache write failed",
                extra={"error": str(exc), "event": "literature.cache.error"},
            )


@dataclass(frozen=True)
class _ProviderBatch:
    papers: list[ExternalPaper]
    stat: ProviderSearchStat


def _aggregate_provider_stats(stats: list[ProviderSearchStat]) -> list[ProviderSearchStat]:
    by_provider: dict[str, ProviderSearchStat] = {}
    for stat in stats:
        existing = by_provider.get(stat.provider)
        if existing is None:
            by_provider[stat.provider] = stat
            continue
        by_provider[stat.provider] = ProviderSearchStat(
            provider=stat.provider,
            count=existing.count + stat.count,
            error=existing.error or stat.error,
        )
    return list(by_provider.values())
