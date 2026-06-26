import pytest

from research_shared.config.settings import Settings
from research_shared.domain.models import Citation, ResearchChunk, SearchResult, SearchType
from research_shared.rag.service import RagService
from research_shared.agents.tools.local_search import local_hybrid_search, merge_local_search_results


class _FakeSearcher:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.last_query = None

    async def search(self, query):
        self.last_query = query
        return self._results


def _result(
    *,
    research_id: str = "r1",
    title: str = "Paper One",
    text: str = "relevant body",
    page: int = 1,
    score: float = 0.83,
) -> SearchResult:
    return SearchResult(
        chunk=ResearchChunk(
            research_id=research_id,
            title=title,
            text=text,
            metadata={"page": page},
        ),
        score=score,
        search_type=SearchType.HYBRID,
    )


@pytest.mark.asyncio
async def test_local_search_builds_numbered_context() -> None:
    results = [
        _result(research_id="r1", title="First", text="Body one", page=1),
        _result(research_id="r2", title="Second", text="Body two", page=2),
    ]
    searcher = _FakeSearcher(results)
    rag = RagService(searcher)

    local = await local_hybrid_search(searcher, rag, "query", limit=5)

    assert "[1]" in local.context
    assert "[2]" in local.context
    assert "First" in local.context
    assert "Second" in local.context
    assert len(local.citations) == 2


@pytest.mark.asyncio
async def test_local_search_empty_results() -> None:
    searcher = _FakeSearcher([])
    rag = RagService(searcher)

    local = await local_hybrid_search(searcher, rag, "query", limit=5)

    assert local.context == ""
    assert local.citations == []
    assert local.results == []


@pytest.mark.asyncio
async def test_local_search_dedupes_citations() -> None:
    results = [
        _result(research_id="r1", title="Paper", text="A", page=3, score=0.9),
        _result(research_id="r1", title="Paper", text="B", page=3, score=0.7),
    ]
    searcher = _FakeSearcher(results)
    rag = RagService(searcher)

    local = await local_hybrid_search(searcher, rag, "query", limit=5)

    assert len(local.citations) == 1
    assert local.context.count("[1]") == 1


@pytest.mark.asyncio
async def test_local_search_context_matches_rag_service() -> None:
    results = [_result(text="shared body", page=4)]
    searcher = _FakeSearcher(results)
    rag = RagService(searcher)
    citations = [
        Citation(
            research_id="r1",
            title="Paper One",
            page=4,
            score=0.83,
        )
    ]

    local = await local_hybrid_search(searcher, rag, "query", limit=3)
    expected = RagService._build_context(results, citations)

    assert local.context == expected


def test_merge_local_search_results_dedupes_overlapping_chunks() -> None:
    shared = _result(research_id="r1", title="Shared", text="overlap", page=1, score=0.9)
    unique = _result(research_id="r2", title="Other", text="unique", page=2, score=0.8)
    low_score = _result(research_id="r1", title="Shared", text="overlap", page=1, score=0.4)

    from research_shared.agents.tools.local_search import LocalSearchResult

    first = LocalSearchResult(
        results=[shared],
        citations=[],
        context="[1] shared",
    )
    second = LocalSearchResult(
        results=[low_score, unique],
        citations=[],
        context="ignored",
    )

    merged = merge_local_search_results([first, second])

    assert len(merged.results) == 2
    assert merged.results[0].chunk.research_id == "r1"
    assert merged.results[0].score == 0.9
    assert merged.results[1].chunk.research_id == "r2"
