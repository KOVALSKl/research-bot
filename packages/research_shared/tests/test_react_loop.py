import json

import pytest

from research_shared.agents.models import AgentReasoningEvent
from research_shared.agents.react_loop import (
    ReactLoopRunner,
    parse_react_output,
    run_rule_based_fallback,
)
from research_shared.agents.state import AgentState
from research_shared.config.settings import Settings
from research_shared.domain.models import Citation, ResearchChunk, SearchResult, SearchType
from research_shared.literature.models import ExternalPaper
from research_shared.rag.service import RagService


class _FakeSearcher:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.call_count = 0

    async def search(self, query):
        self.call_count += 1
        return self._results


class _FakeLiteratureService:
    def __init__(self, papers: list[ExternalPaper] | None = None) -> None:
        self.papers = papers or []
        self.call_count = 0

    async def search_external_queries(self, queries, limit=None, year_from=None, **kwargs):
        self.call_count += 1
        return type(
            "Outcome",
            (),
            {
                "papers": self.papers,
                "provider_stats": (),
                "cache_hit": False,
                "queries": tuple(queries),
            },
        )()

    async def search_with_diagnostics(self, query: str, limit=None, year_from=None, **kwargs):
        self.call_count += 1
        return type(
            "Outcome",
            (),
            {
                "papers": self.papers,
                "provider_stats": (),
                "cache_hit": False,
                "queries": (query,),
            },
        )()


class _SequenceLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._system_prompt = "default"
        self.calls = 0

    def generate(self, question: str, context: str) -> str:
        self.calls += 1
        if not self._responses:
            return '{"thought":"done","action":"finish","action_input":{}}'
        return self._responses.pop(0)


def _result(score: float, text: str = "chunk body text") -> SearchResult:
    return SearchResult(
        chunk=ResearchChunk(research_id="r1", title="Paper", text=text),
        score=score,
        search_type=SearchType.HYBRID,
    )


def _runner(
    *,
    results: list[SearchResult],
    llm: _SequenceLLM,
    literature: _FakeLiteratureService | None = None,
    settings: Settings | None = None,
) -> tuple[ReactLoopRunner, _FakeLiteratureService]:
    settings = settings or Settings(
        _env_file=None,
        agent_relevance_filter_enabled=False,
        agent_max_iterations=6,
    )
    searcher = _FakeSearcher(results)
    rag = RagService(searcher, llm)
    literature = literature or _FakeLiteratureService(
        [
            ExternalPaper(
                title="External paper",
                authors=["Alice"],
                year=2024,
                abstract="External abstract with details.",
                url="https://example.org",
                source="openalex",
            )
        ]
    )
    runner = ReactLoopRunner(
        hybrid_search=searcher,
        rag_service=rag,
        literature_service=literature,
        llm=llm,
        settings=settings,
    )
    return runner, literature


def test_parse_react_output_json() -> None:
    raw = json.dumps(
        {
            "thought": "Need local search",
            "action": "local_hybrid_search",
            "action_input": {"queries": ["test query"]},
        }
    )
    action = parse_react_output(raw)
    assert action is not None
    assert action.action == "local_hybrid_search"
    assert action.thought == "Need local search"


def test_parse_react_output_tagged_fallback() -> None:
    raw = (
        "Thought: Try external\n"
        'Action: external_literature_search\n'
        'Action Input: {"queries": ["ml research"]}'
    )
    action = parse_react_output(raw)
    assert action is not None
    assert action.action == "external_literature_search"


@pytest.mark.asyncio
async def test_react_local_then_finish() -> None:
    llm = _SequenceLLM(
        [
            json.dumps(
                {
                    "thought": "Search locally first",
                    "action": "local_hybrid_search",
                    "action_input": {"query": "question"},
                }
            ),
            json.dumps({"thought": "Enough context", "action": "finish", "action_input": {}}),
        ]
    )
    runner, literature = _runner(results=[_result(0.9), _result(0.8), _result(0.7)], llm=llm)
    state = AgentState(
        message="question?",
        mode="question",
        max_iterations=6,
        search_queries=["question?"],
    )
    reasoning: list[AgentReasoningEvent] = []

    async def on_reasoning(event: AgentReasoningEvent) -> None:
        reasoning.append(event)

    result = await runner.run(state, limit=10, on_reasoning=on_reasoning)

    assert result.state.finished
    assert len(result.state.local_citations) == 1
    assert literature.call_count == 0
    tools = [step.tool for step in result.state.steps]
    assert "local_hybrid_search" in tools
    assert any(event.thought == "Search locally first" for event in reasoning)
    assert reasoning[0].action == "local_hybrid_search"


@pytest.mark.asyncio
async def test_react_local_then_external_then_finish() -> None:
    llm = _SequenceLLM(
        [
            json.dumps(
                {
                    "thought": "Local first",
                    "action": "local_hybrid_search",
                    "action_input": {"query": "question"},
                }
            ),
            json.dumps(
                {
                    "thought": "Need external papers",
                    "action": "external_literature_search",
                    "action_input": {"queries": ["english query"]},
                }
            ),
            json.dumps({"thought": "Ready", "action": "finish", "action_input": {}}),
        ]
    )
    runner, literature = _runner(results=[_result(0.2)], llm=llm)
    state = AgentState(
        message="question?",
        mode="question",
        max_iterations=6,
        search_queries=["question?"],
    )

    result = await runner.run(state, limit=10)

    assert literature.call_count == 1
    assert len(result.state.external_papers) == 1
    assert "external_literature_search" in [s.tool for s in result.state.steps]


@pytest.mark.asyncio
async def test_react_invalid_action_retries() -> None:
    llm = _SequenceLLM(
        [
            json.dumps(
                {
                    "thought": "Bad tool",
                    "action": "unknown_tool",
                    "action_input": {},
                }
            ),
            json.dumps({"thought": "Finish now", "action": "finish", "action_input": {}}),
        ]
    )
    runner, _ = _runner(results=[_result(0.9)], llm=llm)
    state = AgentState(
        message="question?",
        mode="question",
        max_iterations=6,
        search_queries=["question?"],
    )

    result = await runner.run(state, limit=10)

    assert result.state.finished
    assert "Unknown action" in result.state.scratchpad


@pytest.mark.asyncio
async def test_react_max_iterations_forces_finish() -> None:
    llm = _SequenceLLM(
        [
            json.dumps(
                {
                    "thought": "Keep searching",
                    "action": "local_hybrid_search",
                    "action_input": {"query": "q"},
                }
            )
        ]
        * 3
    )
    settings = Settings(
        _env_file=None,
        agent_relevance_filter_enabled=False,
        agent_max_iterations=2,
    )
    searcher = _FakeSearcher([_result(0.9)])
    rag = RagService(searcher, llm)
    runner = ReactLoopRunner(
        hybrid_search=searcher,
        rag_service=rag,
        literature_service=_FakeLiteratureService(),
        llm=llm,
        settings=settings,
    )
    state = AgentState(
        message="question?",
        mode="question",
        max_iterations=2,
        search_queries=["question?"],
    )

    result = await runner.run(state, limit=10)

    assert result.forced_finish
    assert state.iteration == 2


@pytest.mark.asyncio
async def test_react_idea_mode_reformulate_queries() -> None:
    llm = _SequenceLLM(
        [
            json.dumps(
                {
                    "thought": "Reformulate for idea",
                    "action": "reformulate_queries",
                    "action_input": {},
                }
            ),
            json.dumps({"thought": "Done", "action": "finish", "action_input": {}}),
        ]
    )
    runner, _ = _runner(results=[_result(0.9)], llm=llm)
    state = AgentState(
        message="Evaluate my GNN idea",
        mode="idea_evaluation",
        max_iterations=6,
        search_queries=["Evaluate my GNN idea"],
    )

    result = await runner.run(state, limit=10)

    reform = next(s for s in result.state.steps if s.tool == "reformulate_queries")
    assert reform.results_count >= 1


@pytest.mark.asyncio
async def test_rule_based_fallback_llm_disabled() -> None:
    searcher = _FakeSearcher([_result(0.9, text="Local snippet")])
    rag = RagService(searcher, None)
    state = AgentState(
        message="question?",
        mode="question",
        max_iterations=6,
        search_queries=["question?"],
    )

    result = await run_rule_based_fallback(
        state,
        hybrid_search=searcher,
        rag_service=rag,
        limit=10,
    )

    assert result.finished
    assert len(result.local_citations) == 1
    assert result.steps[0].detail == "llm=disabled"


@pytest.mark.asyncio
async def test_ensure_idea_external_search_mandatory_when_weak_local() -> None:
    literature = _FakeLiteratureService(
        papers=[
            ExternalPaper(
                title="External Paper",
                url="https://example.org/paper",
                pdf_url="https://example.org/paper.pdf",
                source="openalex",
                abstract="Abstract",
            )
        ]
    )
    runner, _ = _runner(
        results=[],
        llm=_SequenceLLM([]),
        literature=literature,
        settings=Settings(_env_file=None, agent_idea_min_local_chunks=1),
    )
    state = AgentState(
        message="Evaluate idea",
        mode="idea_evaluation",
        max_iterations=6,
        search_queries=["Evaluate idea"],
    )

    await runner.ensure_idea_external_search(state, limit=5)

    assert literature.call_count == 1
    assert state.external_papers
    assert state.steps[-1].tool == "external_literature_search"
    assert "mandatory" in (state.steps[-1].detail or "")
