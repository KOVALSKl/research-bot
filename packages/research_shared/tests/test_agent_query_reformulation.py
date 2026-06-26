import pytest

from research_shared.agents.models import AgentProgressStage
from research_shared.agents.query_reformulation import (
    build_idea_search_queries,
    build_search_queries,
    reformulate_question,
)
from research_shared.config.settings import Settings


class _FakeLLM:
    def __init__(self, answer: str) -> None:
        self._system_prompt = "default"
        self.answer = answer
        self.calls = 0

    def generate(self, question: str, context: str) -> str:
        self.calls += 1
        return self.answer


@pytest.mark.asyncio
async def test_reformulation_disabled_returns_original() -> None:
    settings = Settings(_env_file=None, agent_query_reformulation_enabled=False)
    llm = _FakeLLM("reformulated query")

    result = await reformulate_question("original?", llm, settings)

    assert result == "original?"
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_reformulation_mock_llm() -> None:
    settings = Settings(_env_file=None, agent_query_reformulation_enabled=True)
    llm = _FakeLLM("reformulated EN query")

    result = await reformulate_question("долги физлиц", llm, settings)

    assert result == "reformulated EN query"
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_reformulation_llm_error_fallback() -> None:
    settings = Settings(_env_file=None, agent_query_reformulation_enabled=True)

    class _BrokenLLM:
        _system_prompt = "x"

        def generate(self, question: str, context: str) -> str:
            raise RuntimeError("fail")

    result = await reformulate_question("original?", _BrokenLLM(), settings)

    assert result == "original?"


@pytest.mark.asyncio
async def test_build_search_queries_disabled_returns_original() -> None:
    settings = Settings(_env_file=None, agent_query_reformulation_enabled=False)
    llm = _FakeLLM("ignored")

    queries = await build_search_queries("original?", llm, settings)

    assert queries == ["original?"]
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_build_search_queries_multiline_parse() -> None:
    settings = Settings(_env_file=None, agent_query_reformulation_enabled=True)
    llm = _FakeLLM(
        "долговые обязательства бот\n"
        "debt obligations financial planning chatbot research"
    )

    queries = await build_search_queries("есть ли исследования?", llm, settings)

    assert queries[0] == "есть ли исследования?"
    assert len(queries) == 3
    assert "debt obligations" in queries[-1]


@pytest.mark.asyncio
async def test_build_search_queries_respects_max_limit() -> None:
    settings = Settings(
        _env_file=None,
        agent_query_reformulation_enabled=True,
        agent_search_queries_max=2,
    )
    llm = _FakeLLM("query ru\nquery en\nquery extra")

    queries = await build_search_queries("original", llm, settings)

    assert len(queries) == 2
    assert queries[0] == "original"


@pytest.mark.asyncio
async def test_build_idea_search_queries_uses_idea_prompt() -> None:
    settings = Settings(
        _env_file=None,
        agent_idea_query_reformulation_enabled=True,
        agent_search_queries_max=4,
    )
    llm = _FakeLLM("idea normalized\nрусский запрос по идее\nenglish idea query")

    queries = await build_idea_search_queries(
        "Насколько перспективна идея мембранных материалов?",
        llm,
        settings,
    )

    assert queries[0].startswith("Насколько перспективна")
    assert len(queries) >= 3
    assert "english idea query" in queries


@pytest.mark.asyncio
async def test_build_idea_search_queries_aspect_fallback_without_llm() -> None:
    settings = Settings(_env_file=None, agent_idea_query_reformulation_enabled=True)
    queries = await build_idea_search_queries("GNN для fraud detection", None, settings)
    assert len(queries) >= 2
    assert queries[0] == "GNN для fraud detection"
