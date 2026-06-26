import pytest

from research_shared.agents.tools.external_search import (
    external_literature_search,
    external_literature_search_queries,
)
from research_shared.literature.models import ExternalPaper
from research_shared.literature.service import ExternalLiteratureSearchOutcome, ProviderSearchStat


class _FakeLiteratureService:
    def __init__(self, papers: list[ExternalPaper]) -> None:
        self.papers = papers
        self.last_limit: int | None = None
        self.last_queries: list[str] | None = None

    async def search_external(self, query: str, limit=None, year_from=None):
        self.last_limit = limit
        return self.papers

    async def search_external_queries(self, queries, limit=None, year_from=None, *, mode=None):
        self.last_limit = limit
        self.last_queries = list(queries)
        return ExternalLiteratureSearchOutcome(
            papers=self.papers,
            provider_stats=(ProviderSearchStat(provider="stub", count=len(self.papers)),),
            queries=tuple(queries),
        )

    async def search_with_diagnostics(self, query: str, limit=None, year_from=None, *, mode=None):
        self.last_queries = [query]
        return ExternalLiteratureSearchOutcome(
            papers=self.papers,
            provider_stats=(ProviderSearchStat(provider="stub", count=len(self.papers)),),
            queries=(query,),
        )


def _paper(title: str, index: int) -> ExternalPaper:
    return ExternalPaper(
        title=title,
        authors=[f"Author {index}"],
        year=2020 + index,
        abstract=f"Abstract {index}.",
        doi=f"10.1234/{index}",
        url=f"https://example.org/{index}",
        source="openalex",
    )


@pytest.mark.asyncio
async def test_external_search_builds_en_context() -> None:
    service = _FakeLiteratureService([_paper("Paper A", 1), _paper("Paper B", 2)])

    result = await external_literature_search(service, "query", limit=5)

    assert "[E1]" in result.context
    assert "[E2]" in result.context
    assert "Abstract 1." in result.context
    assert len(result.papers) == 2


@pytest.mark.asyncio
async def test_external_search_empty_list() -> None:
    service = _FakeLiteratureService([])

    result = await external_literature_search(service, "query", limit=5)

    assert result.context == ""
    assert result.papers == []


@pytest.mark.asyncio
async def test_external_search_passes_limit() -> None:
    service = _FakeLiteratureService([_paper("Paper A", 1)])

    await external_literature_search(service, "query", limit=7)

    assert service.last_limit == 7


@pytest.mark.asyncio
async def test_external_search_multi_query_fallback() -> None:
    class _EmptyThenFallbackService:
        def __init__(self) -> None:
            self.calls = 0

        async def search_external_queries(self, queries, limit=None, year_from=None, *, mode=None):
            self.calls += 1
            return ExternalLiteratureSearchOutcome(papers=[], queries=tuple(queries))

        async def search_with_diagnostics(self, query: str, limit=None, year_from=None, *, mode=None):
            self.calls += 1
            return ExternalLiteratureSearchOutcome(
                papers=[_paper("Fallback", 1)],
                queries=(query,),
            )

    service = _EmptyThenFallbackService()

    result = await external_literature_search_queries(
        service,
        ["en query"],
        limit=5,
        fallback_query="original question",
    )

    assert result.fallback_used is True
    assert len(result.papers) == 1
    assert service.calls == 2
