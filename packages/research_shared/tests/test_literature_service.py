import fakeredis.aioredis
import pytest

from research_shared.config.settings import Settings
from research_shared.literature.models import ExternalPaper
from research_shared.literature.service import ExternalLiteratureService, _cache_key_queries, _dedupe_papers


class _StubProvider:
    def __init__(self, name: str, papers: list[ExternalPaper] | None = None, *, fail: bool = False) -> None:
        self.name = name
        self._papers = papers or []
        self._fail = fail
        self.calls = 0

    async def search(self, query: str, limit: int, year_from: int | None = None) -> list[ExternalPaper]:
        self.calls += 1
        if self._fail:
            raise RuntimeError(f"{self.name} failed")
        return self._papers[:limit]


def _paper(title: str, *, doi: str | None = None, source: str = "openalex") -> ExternalPaper:
    return ExternalPaper(
        title=title,
        authors=["Author"],
        year=2023,
        abstract="abstract",
        doi=doi,
        url=f"https://example.org/{title.replace(' ', '-').lower()}",
        source=source,
    )


def test_dedupe_by_doi() -> None:
    papers = [
        _paper("Paper A", doi="10.1234/abc", source="openalex"),
        _paper("Paper A copy", doi="10.1234/abc", source="arxiv"),
        _paper("Paper B", doi="10.9999/xyz", source="arxiv"),
    ]
    deduped = _dedupe_papers(papers)
    assert len(deduped) == 2
    assert deduped[0].source == "openalex"


def test_dedupe_by_normalized_title_without_doi() -> None:
    papers = [
        _paper("Graph Neural Networks!", doi=None, source="openalex"),
        _paper("graph neural networks", doi=None, source="arxiv"),
        _paper("Different Paper", doi=None, source="arxiv"),
    ]
    deduped = _dedupe_papers(papers)
    assert len(deduped) == 2


@pytest.mark.asyncio
async def test_service_merge_and_limit() -> None:
    settings = Settings(_env_file=None, literature_default_limit=2)
    providers = [
        _StubProvider("a", [_paper("One", doi="10.1/a"), _paper("Two", doi="10.1/b")]),
        _StubProvider("b", [_paper("Three", doi="10.1/c")]),
    ]
    service = ExternalLiteratureService(settings, providers)

    papers = await service.search_external("query")

    assert len(papers) == 2
    assert providers[0].calls == 1
    assert providers[1].calls == 1


@pytest.mark.asyncio
async def test_service_partial_failure() -> None:
    settings = Settings(_env_file=None)
    good = _StubProvider("good", [_paper("OK", doi="10.1/ok")])
    bad = _StubProvider("bad", fail=True)
    service = ExternalLiteratureService(settings, [good, bad])

    papers = await service.search_external("query")

    assert len(papers) == 1
    assert papers[0].title == "OK"


@pytest.mark.asyncio
async def test_service_dedupes_duplicate_doi_across_providers() -> None:
    settings = Settings(_env_file=None, literature_default_limit=10)
    providers = [
        _StubProvider("openalex", [_paper("GNN Paper", doi="10.1234/gnn.fin.2023", source="openalex")]),
        _StubProvider("semantic", [_paper("GNN Paper SS", doi="10.1234/gnn.fin.2023", source="semantic_scholar")]),
    ]
    service = ExternalLiteratureService(settings, providers)

    papers = await service.search_external("gnn")

    assert len(papers) == 1
    assert papers[0].source == "openalex"


@pytest.mark.asyncio
async def test_service_cache_miss_then_hit() -> None:
    settings = Settings(_env_file=None, literature_cache_ttl_seconds=3600)
    provider = _StubProvider("p", [_paper("Cached", doi="10.1/cache")])
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    service = ExternalLiteratureService(settings, [provider], redis_client=redis_client)

    first = await service.search_external("cache query", limit=5)
    second = await service.search_external("cache query", limit=5)

    assert len(first) == 1
    assert len(second) == 1
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_service_cache_respects_different_keys() -> None:
    settings = Settings(_env_file=None)
    provider = _StubProvider("p", [_paper("X", doi="10.1/x")])
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    service = ExternalLiteratureService(settings, [provider], redis_client=redis_client)

    await service.search_external("query-a", limit=5)
    await service.search_external("query-b", limit=5)

    assert provider.calls == 2


@pytest.mark.asyncio
async def test_service_without_redis_always_calls_providers() -> None:
    settings = Settings(_env_file=None)
    provider = _StubProvider("p", [_paper("No cache")])
    service = ExternalLiteratureService(settings, [provider], redis_client=None)

    await service.search_external("q")
    await service.search_external("q")

    assert provider.calls == 2


@pytest.mark.asyncio
async def test_service_uses_default_limit_from_settings() -> None:
    settings = Settings(_env_file=None, literature_default_limit=1)
    provider = _StubProvider(
        "p",
        [_paper("One", doi="10.1/1"), _paper("Two", doi="10.1/2")],
    )
    service = ExternalLiteratureService(settings, [provider])

    papers = await service.search_external("q")

    assert len(papers) == 1


@pytest.mark.asyncio
async def test_service_empty_results_use_short_cache_ttl() -> None:
    settings = Settings(
        _env_file=None,
        literature_cache_ttl_seconds=3600,
        literature_cache_empty_ttl_seconds=30,
    )
    provider = _StubProvider("p", [])
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    service = ExternalLiteratureService(settings, [provider], redis_client=redis_client)

    await service.search_external("empty query", limit=5)

    keys = await redis_client.keys("literature:*")
    assert len(keys) == 1
    ttl = await redis_client.ttl(keys[0])
    assert 0 < ttl <= 30


@pytest.mark.asyncio
async def test_service_search_with_diagnostics_returns_provider_stats() -> None:
    settings = Settings(_env_file=None)
    good = _StubProvider("good", [_paper("OK", doi="10.1/ok")])
    bad = _StubProvider("bad", fail=True)
    service = ExternalLiteratureService(settings, [good, bad])

    outcome = await service.search_with_diagnostics("query")

    assert len(outcome.papers) == 1
    assert len(outcome.provider_stats) == 2
    errors = [stat.error for stat in outcome.provider_stats if stat.error]
    assert errors


def test_multi_query_cache_keys_differ() -> None:
    key_a = _cache_key_queries(["query-a"], 10, None)
    key_b = _cache_key_queries(["query-a", "query-b"], 10, None)
    assert key_a != key_b


@pytest.mark.asyncio
async def test_service_search_quality_regression() -> None:
    settings = Settings(_env_file=None, literature_default_limit=5)
    providers = [
        _StubProvider(
            "openalex",
            [
                ExternalPaper(
                    title="Financial pyramid modeling",
                    authors=["Alice"],
                    year=2024,
                    abstract="Modeling financial pyramid activity with citations.",
                    doi="10.1/pyramid",
                    url="https://example.org/pyramid",
                    pdf_url="https://example.org/pyramid.pdf",
                    source="openalex",
                ),
                ExternalPaper(
                    title="Unrelated topic",
                    authors=["Bob"],
                    year=2020,
                    abstract="",
                    url="https://example.org/other",
                    source="openalex",
                ),
            ],
        )
    ]
    service = ExternalLiteratureService(settings, providers)

    papers = await service.search_external("financial pyramid modeling", limit=5)

    assert papers
    assert any(paper.pdf_url for paper in papers)
    with_abstract = [paper for paper in papers if paper.abstract.strip()]
    assert len(with_abstract) / len(papers) >= 0.8


@pytest.mark.asyncio
async def test_service_multi_query_includes_secondary_query_papers() -> None:
    settings = Settings(_env_file=None, literature_default_limit=5)

    class _QueryAwareProvider:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def search(self, query: str, limit: int, year_from: int | None = None):
            self.queries.append(query)
            if query == "primary query":
                return [_paper("Primary hit", doi="10.1/primary")]
            return [_paper("Secondary hit", doi="10.1/secondary", source="arxiv")]

    provider = _QueryAwareProvider()
    service = ExternalLiteratureService(settings, [provider])

    outcome = await service.search_external_queries(
        ["primary query", "secondary query"],
        limit=5,
    )
    papers = outcome.papers

    assert len(papers) >= 2
    titles = {paper.title for paper in papers}
    assert "Primary hit" in titles
    assert "Secondary hit" in titles
