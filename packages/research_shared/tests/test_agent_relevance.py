import pytest

from research_shared.agents.models import AgentStep
from research_shared.agents.relevance import filter_relevant_results
from research_shared.agents.tools.local_search import LocalSearchResult
from research_shared.config.settings import Settings
from research_shared.domain.models import Citation, ResearchChunk, SearchResult, SearchType


class _FakeLLM:
    def __init__(self, answer: str = "1, 2") -> None:
        self._system_prompt = "default"
        self.answer = answer

    def generate(self, question: str, context: str) -> str:
        return self.answer


def _result(score: float, text: str = "body", title: str = "Chunk") -> SearchResult:
    return SearchResult(
        chunk=ResearchChunk(research_id="r1", title=title, text=text),
        score=score,
        search_type=SearchType.HYBRID,
    )


def _local(*results: SearchResult) -> LocalSearchResult:
    citations = [
        Citation(
            research_id=result.chunk.research_id,
            title=result.chunk.title,
            page=index + 1,
            score=result.score,
        )
        for index, result in enumerate(results)
    ]
    context = "\n\n".join(f"[{index + 1}] {result.chunk.title}" for index, result in enumerate(results))
    return LocalSearchResult(results=list(results), citations=citations, context=context)


@pytest.mark.asyncio
async def test_l1_removes_low_score_chunks() -> None:
    settings = Settings(
        _env_file=None,
        agent_relevance_filter_enabled=True,
        agent_min_chunk_score=0.6,
        agent_use_llm_relevance_filter=False,
    )
    local = _local(_result(0.9), _result(0.4), _result(0.7))

    filtered = await filter_relevant_results("question?", local, settings, llm=None)

    assert len(filtered.results) == 2
    assert filtered.filtered_count == 1


@pytest.mark.asyncio
async def test_l2_mock_llm_filters_indices() -> None:
    settings = Settings(
        _env_file=None,
        agent_relevance_filter_enabled=True,
        agent_min_chunk_score=0.0,
        agent_use_llm_relevance_filter=True,
        llm_enabled=True,
    )
    local = _local(
        _result(0.9, text="personal debt"),
        _result(0.8, text="corporate finance"),
    )

    filtered = await filter_relevant_results(
        "personal debt question",
        local,
        settings,
        llm=_FakeLLM(answer="1"),
    )

    assert len(filtered.results) == 1
    assert "personal debt" in filtered.results[0].chunk.text


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_l1() -> None:
    settings = Settings(
        _env_file=None,
        agent_relevance_filter_enabled=True,
        agent_min_chunk_score=0.6,
        agent_use_llm_relevance_filter=True,
        llm_enabled=True,
    )
    local = _local(_result(0.9), _result(0.4))

    class _BrokenLLM:
        _system_prompt = "x"

        def generate(self, question: str, context: str) -> str:
            raise RuntimeError("boom")

    filtered = await filter_relevant_results(
        "question?",
        local,
        settings,
        llm=_BrokenLLM(),
    )

    assert len(filtered.results) == 1


@pytest.mark.asyncio
async def test_empty_after_filter() -> None:
    settings = Settings(
        _env_file=None,
        agent_relevance_filter_enabled=True,
        agent_min_chunk_score=0.99,
    )
    local = _local(_result(0.5), _result(0.4))

    filtered = await filter_relevant_results("question?", local, settings, llm=None)

    assert filtered.results == []
    assert filtered.filtered_count == 2


@pytest.mark.asyncio
async def test_filter_disabled_is_noop() -> None:
    settings = Settings(_env_file=None, agent_relevance_filter_enabled=False)
    local = _local(_result(0.1))

    filtered = await filter_relevant_results("question?", local, settings, llm=None)

    assert len(filtered.results) == 1
    assert filtered.filtered_count == 0
