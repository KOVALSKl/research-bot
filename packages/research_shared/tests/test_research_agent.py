import json

import pytest

from research_shared.agents.models import AgentAskRequest, AgentProgressStage, AgentReasoningEvent
from research_shared.agents.research_agent import ResearchAgent
from research_shared.config.settings import Settings
from research_shared.domain.models import ResearchChunk, SearchResult, SearchType
from research_shared.literature.models import ExternalPaper
from research_shared.rag.service import RagService

_LONG_ANSWER = (
    "Detailed synthesized answer about debt repayment planning strategies found in "
    "local materials, including avalanche method prioritization and concrete savings "
    "estimates from the cited research literature source [1] and external review [E1]."
)


def _react_action(thought: str, action: str, **action_input: object) -> str:
    return json.dumps(
        {"thought": thought, "action": action, "action_input": action_input},
        ensure_ascii=False,
    )


class _FakeSearcher:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def search(self, query):
        return self._results


class _FakeLiteratureService:
    def __init__(self, papers: list[ExternalPaper] | None = None) -> None:
        self.papers = papers or []
        self.call_count = 0
        self.last_queries: list[str] | None = None

    async def search_external_queries(self, queries, limit=None, year_from=None, **kwargs):
        self.call_count += 1
        self.last_queries = list(queries)
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
        self.last_queries = [query]
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


class _ReactSequenceLLM:
    def __init__(self, react_responses: list[str], synthesis: str) -> None:
        self._react = list(react_responses)
        self._synthesis = synthesis
        self._system_prompt = "default"
        self.calls = 0

    def generate(self, question: str, context: str) -> str:
        self.calls += 1
        if self._react and ("Choose the next action" in question or "Scratchpad" in question):
            return self._react.pop(0)
        return self._synthesis


def _result(score: float, text: str = "body") -> SearchResult:
    return SearchResult(
        chunk=ResearchChunk(research_id="r1", title="Paper", text=text),
        score=score,
        search_type=SearchType.HYBRID,
    )


def _agent(
    *,
    results: list[SearchResult],
    settings: Settings | None = None,
    llm: _ReactSequenceLLM | None = None,
    literature: _FakeLiteratureService | None = None,
) -> tuple[ResearchAgent, _FakeLiteratureService]:
    settings = settings or Settings(
        _env_file=None,
        min_local_results=3,
        min_local_score=0.5,
        agent_relevance_filter_enabled=False,
        agent_idea_query_reformulation_enabled=False,
    )
    searcher = _FakeSearcher(results)
    rag = RagService(searcher, llm)
    literature = literature or _FakeLiteratureService(
        [
            ExternalPaper(
                title="External",
                authors=["Alice"],
                year=2024,
                abstract="External abstract.",
                url="https://example.org",
                source="openalex",
            )
        ]
    )
    agent = ResearchAgent(
        hybrid_search=searcher,
        rag_service=rag,
        literature_service=literature,
        llm_provider=llm,
        settings=settings,
    )
    return agent, literature


@pytest.mark.asyncio
async def test_react_local_sufficient_skips_external() -> None:
    llm = _ReactSequenceLLM(
        [
            _react_action("Search PDFs", "local_hybrid_search", query="question"),
            _react_action("Ready", "finish"),
        ],
        _LONG_ANSWER,
    )
    agent, literature = _agent(
        results=[_result(0.9), _result(0.8), _result(0.7)],
        llm=llm,
    )

    response = await agent.run(AgentAskRequest(message="question?", mode="question"))

    assert literature.call_count == 0
    tools = [step.tool for step in response.steps]
    assert tools[0] == "classify"
    assert "local_hybrid_search" in tools
    assert "external_literature_search" not in tools
    assert "[1]" in response.answer


@pytest.mark.asyncio
async def test_react_local_insufficient_calls_external() -> None:
    llm = _ReactSequenceLLM(
        [
            _react_action("Local", "local_hybrid_search", query="question"),
            _react_action("External", "external_literature_search", queries=["english query"]),
            _react_action("Done", "finish"),
        ],
        _LONG_ANSWER,
    )
    agent, literature = _agent(results=[_result(0.3)], llm=llm)

    response = await agent.run(AgentAskRequest(message="question?", mode="auto"))

    assert literature.call_count == 1
    assert "external_literature_search" in [step.tool for step in response.steps]
    assert len(response.sources.external) == 1


@pytest.mark.asyncio
async def test_react_llm_disabled_fallback() -> None:
    results = [_result(0.9, text="Local snippet text")]
    agent, _ = _agent(results=results, llm=None)

    response = await agent.run(AgentAskRequest(message="question?", mode="question"))

    assert "LLM отключён" in response.answer
    assert "Local snippet" in response.answer


@pytest.mark.asyncio
async def test_react_steps_have_results_count() -> None:
    llm = _ReactSequenceLLM(
        [
            _react_action("Search", "local_hybrid_search", query="question"),
            _react_action("Done", "finish"),
        ],
        _LONG_ANSWER,
    )
    agent, _ = _agent(
        results=[_result(0.9), _result(0.8), _result(0.7)],
        llm=llm,
    )

    response = await agent.run(AgentAskRequest(message="question?", mode="question"))

    local_step = next(step for step in response.steps if step.tool == "local_hybrid_search" and step.thought == "Search")
    assert local_step.results_count == 3
    assert local_step.thought == "Search"
    assert response.mode == "question"


@pytest.mark.asyncio
async def test_react_progress_callback_emits_classify_and_synthesize() -> None:
    llm = _ReactSequenceLLM([_react_action("Done", "finish")], _LONG_ANSWER)
    agent, _ = _agent(results=[_result(0.9)], llm=llm)
    events = []

    async def on_progress(event):
        events.append(event.stage)

    await agent.run(AgentAskRequest(message="question?", mode="question"), on_progress=on_progress)

    assert events == [AgentProgressStage.CLASSIFY, AgentProgressStage.SYNTHESIZE]


@pytest.mark.asyncio
async def test_react_reasoning_callback_emits_events() -> None:
    llm = _ReactSequenceLLM(
        [
            _react_action("Search", "local_hybrid_search", query="question"),
            _react_action("Done", "finish"),
        ],
        _LONG_ANSWER,
    )
    agent, _ = _agent(results=[_result(0.9)], llm=llm)
    events: list[AgentReasoningEvent] = []

    async def on_reasoning(event: AgentReasoningEvent) -> None:
        events.append(event)

    await agent.run(
        AgentAskRequest(message="question?", mode="question"),
        on_reasoning=on_reasoning,
    )

    assert len(events) >= 2
    assert any(event.thought == "Search" for event in events)


class _RetryFakeLLM:
    def __init__(self) -> None:
        self._system_prompt = "default"
        self.synthesis_calls = 0
        self.react_step = 0

    def generate(self, question: str, context: str) -> str:
        if "Choose the next action" in question or "Scratchpad" in question:
            if self.react_step == 0:
                self.react_step += 1
                return _react_action("Search", "local_hybrid_search", query="question")
            return _react_action("Done", "finish")
        self.synthesis_calls += 1
        if self.synthesis_calls == 1:
            return "Короткий ответ без ссылок."
        return _LONG_ANSWER


@pytest.mark.asyncio
async def test_react_synthesis_retry_on_shallow_answer() -> None:
    settings = Settings(
        _env_file=None,
        agent_relevance_filter_enabled=False,
        agent_synthesis_retry_on_shallow=True,
        agent_query_reformulation_enabled=False,  # Disable to avoid extra LLM calls in this test
    )
    llm = _RetryFakeLLM()
    agent, _ = _agent(results=[_result(0.9), _result(0.8), _result(0.7)], settings=settings, llm=llm)

    response = await agent.run(AgentAskRequest(message="question?", mode="question"))

    assert llm.synthesis_calls == 2
    synth_step = next(step for step in response.steps if step.tool == "synthesize")
    assert synth_step.detail.startswith("retry=true")
    assert "[1]" in response.answer


@pytest.mark.asyncio
async def test_react_extracts_cited_sources_and_source_files() -> None:
    llm = _ReactSequenceLLM(
        [
            _react_action("Local", "local_hybrid_search", query="question"),
            _react_action("External", "external_literature_search", queries=["en query"]),
            _react_action("Done", "finish"),
        ],
        "Detailed answer citing local finding [1] and external paper conclusion [E1]. " * 4,
    )
    agent, _ = _agent(results=[_result(0.3)], llm=llm)

    response = await agent.run(AgentAskRequest(message="question?", mode="auto"))

    assert len(response.sources.local) == 1
    assert response.sources.local_indices == [1]
    assert len(response.sources.external) == 1
    assert len(response.source_files) == 1


@pytest.mark.asyncio
async def test_react_reformulate_queries_step() -> None:
    settings = Settings(
        _env_file=None,
        agent_relevance_filter_enabled=False,
        agent_query_reformulation_enabled=True,
    )
    llm = _ReactSequenceLLM(
        [
            _react_action("Reformulate", "reformulate_queries"),
            _react_action("Search", "local_hybrid_search", query="english research query"),
            _react_action("External", "external_literature_search", queries=["english research query"]),
            _react_action("Done", "finish"),
        ],
        _LONG_ANSWER,
    )
    agent, literature = _agent(results=[_result(0.3)], settings=settings, llm=llm)

    response = await agent.run(AgentAskRequest(message="есть исследования?", mode="question"))

    reform_step = next(step for step in response.steps if step.tool == "reformulate_queries")
    assert reform_step.results_count >= 1
    assert literature.call_count == 1


_IDEA_JSON = """\
{
  "relevance": "Идея связана с существующими работами по графовым методам анализа транзакций [1].",
  "evidence_for": ["Графовые нейросети успешно выявляют мошеннические паттерны транзакций с высокой точностью на публичных датасетах [1]."],
  "evidence_against": ["Во внешней литературе отмечены ограничения GNN при разреженных графах и неполных данных [E1]."],
  "success_outlook": "Умеренные перспективы при наличии качественных графовых представлений транзакций [E1].",
  "confidence": "medium",
  "summary": "Идея имеет основания в локальных материалах [1], но требует учёта ограничений из [E1]."
}"""


class _IdeaEvalLLM:
    def __init__(self, *, include_external: bool = True) -> None:
        self._system_prompt = "default"
        self.react_step = 0
        self.include_external = include_external
        self.calls: list[tuple[str, str, str]] = []

    def generate(self, question: str, context: str) -> str:
        if "Choose the next action" in question or "Scratchpad" in question:
            if self.react_step == 0:
                self.react_step += 1
                return _react_action("Local", "local_hybrid_search", query="idea")
            if self.include_external and self.react_step == 1:
                self.react_step += 1
                return _react_action("External", "external_literature_search", queries=["idea en"])
            return _react_action("Done", "finish")
        self.calls.append((self._system_prompt, question, context))
        return _IDEA_JSON


@pytest.mark.asyncio
async def test_react_idea_evaluation_returns_assessment() -> None:
    agent, _ = _agent(results=[_result(0.3)], llm=_IdeaEvalLLM(include_external=False))

    response = await agent.run(
        AgentAskRequest(
            message="Оцени идею применения GNN к транзакциям",
            mode="idea_evaluation",
        )
    )

    assert response.mode == "idea_evaluation"
    assert response.idea_assessment is not None
    assert response.idea_assessment.confidence == "medium"
    assert "[1]" in response.answer or "[E1]" in response.answer


@pytest.mark.asyncio
async def test_react_idea_reformulate_queries() -> None:
    settings = Settings(
        _env_file=None,
        agent_relevance_filter_enabled=False,
        agent_query_reformulation_enabled=True,
    )
    llm = _ReactSequenceLLM(
        [
            _react_action("Reformulate idea", "reformulate_queries"),
            _react_action("Local", "local_hybrid_search", query="membrane idea"),
            _react_action("Done", "finish"),
        ],
        _IDEA_JSON,
    )
    agent, _ = _agent(results=[_result(0.3)], settings=settings, llm=llm)

    response = await agent.run(
        AgentAskRequest(
            message="Насколько перспективна идея мембранных материалов?",
            mode="idea_evaluation",
        )
    )

    reform_step = next(step for step in response.steps if step.tool == "reformulate_queries")
    assert "idea →" in (reform_step.detail or "")
    assert response.idea_assessment is not None


@pytest.mark.asyncio
async def test_react_idea_auto_classify_end_to_end() -> None:
    agent, _ = _agent(results=[_result(0.3)], llm=_IdeaEvalLLM(include_external=False))

    response = await agent.run(
        AgentAskRequest(
            message="Насколько перспективна идея мембранных материалов для фильтрации?",
            mode="auto",
        )
    )

    assert response.mode == "idea_evaluation"
    assert response.idea_assessment is not None


@pytest.mark.asyncio
async def test_react_idea_external_with_citations() -> None:
    agent, literature = _agent(results=[_result(0.3)], llm=_IdeaEvalLLM(include_external=True))

    response = await agent.run(
        AgentAskRequest(
            message="Оцени идею применения мембран в водоочистке",
            mode="idea_evaluation",
        )
    )

    assert literature.call_count == 1
    assert "external_literature_search" in [step.tool for step in response.steps]
    assert response.idea_assessment is not None


class _IdeaEvalNoCitationLLM(_IdeaEvalLLM):
    def __init__(self) -> None:
        super().__init__(include_external=False)
        self.synthesis_calls = 0

    def generate(self, question: str, context: str) -> str:
        if "Choose the next action" in question or "Scratchpad" in question:
            return super().generate(question, context)
        self.synthesis_calls += 1
        if self.synthesis_calls == 1:
            return """{
  "relevance": "Relevant topic.",
  "evidence_for": ["Supports the approach."],
  "evidence_against": ["Some limitations exist."],
  "success_outlook": "Moderate outlook.",
  "confidence": "medium",
  "summary": "Summary without citations."
}"""
        return _IDEA_JSON


@pytest.mark.asyncio
async def test_react_idea_retries_when_evidence_shallow() -> None:
    llm = _IdeaEvalNoCitationLLM()
    agent, _ = _agent(results=[_result(0.3)], llm=llm)

    response = await agent.run(
        AgentAskRequest(message="Оцени идею GNN для fraud", mode="idea_evaluation")
    )

    assert llm.synthesis_calls >= 2
    assert response.idea_assessment is not None
    synth_step = next(step for step in response.steps if step.tool == "synthesize")
    assert "retry=" in (synth_step.detail or "")
    assert not ResearchAgent._is_shallow_evidence(response.idea_assessment.evidence_for[0].text)


def test_is_shallow_evidence_rejects_template() -> None:
    assert ResearchAgent._is_shallow_evidence("аргумент за [1] и конкретным фактом")
    assert ResearchAgent._is_shallow_evidence("информация в [E1]")


def test_is_shallow_evidence_accepts_substance() -> None:
    text = (
        "Алгоритмы приоритизации долгов по процентной ставке снижают общую переплату "
        "по сравнению с минимальными платежами [E2]."
    )
    assert not ResearchAgent._is_shallow_evidence(text)


class _ShallowIdeaLLM(_IdeaEvalLLM):
    def __init__(self) -> None:
        super().__init__(include_external=False)
        self.synthesis_calls = 0

    def generate(self, question: str, context: str) -> str:
        if "Choose the next action" in question or "Scratchpad" in question:
            return super().generate(question, context)
        self.synthesis_calls += 1
        if self.synthesis_calls == 1:
            return """{
  "relevance": "Relevant topic.",
  "evidence_for": ["аргумент за [1] и конкретным фактом"],
  "evidence_against": ["информация в [E1]"],
  "success_outlook": "Moderate.",
  "confidence": "medium",
  "summary": "Summary."
}"""
        return _IDEA_JSON


@pytest.mark.asyncio
async def test_react_idea_eval_retries_shallow_evidence() -> None:
    llm = _ShallowIdeaLLM()
    agent, _ = _agent(results=[_result(0.3)], llm=llm)

    response = await agent.run(
        AgentAskRequest(message="Оцени идею GNN", mode="idea_evaluation")
    )

    assert response.idea_assessment is not None
    assert llm.synthesis_calls >= 2
    assert not ResearchAgent._is_shallow_evidence(response.idea_assessment.evidence_for[0].text)


@pytest.mark.asyncio
async def test_explicit_question_mode_not_classified_as_idea() -> None:
    llm = _ReactSequenceLLM(
        [
            _react_action("Search", "local_hybrid_search", query="question"),
            _react_action("Done", "finish"),
        ],
        _LONG_ANSWER,
    )
    agent, _ = _agent(results=[_result(0.9)], llm=llm)

    response = await agent.run(
        AgentAskRequest(
            message="Насколько перспективна идея мембранных материалов для фильтрации?",
            mode="question",
        )
    )

    assert response.mode == "question"
    assert response.idea_assessment is None


@pytest.mark.asyncio
async def test_parse_structured_relevance_from_idea_json() -> None:
    payload = ResearchAgent._parse_relevance(
        {
            "level": "high",
            "criteria": [
                {
                    "name": "local_sources",
                    "level": "medium",
                    "detail": "Есть локальные данные [1].",
                }
            ],
            "rationale": "Итог высокий [1].",
        }
    )
    assert payload.level == "high"
    assert payload.criteria[0].name == "local_sources"


@pytest.mark.asyncio
async def test_react_finish_without_search_is_rejected() -> None:
    llm = _ReactSequenceLLM([_react_action("Done", "finish")], _LONG_ANSWER)
    agent, literature = _agent(
        results=[],
        llm=llm,
        literature=_FakeLiteratureService(papers=[]),
        settings=Settings(_env_file=None, llm_enabled=True, agent_relevance_filter_enabled=False),
    )

    response = await agent.run(AgentAskRequest(message="question?", mode="question"))

    assert literature.call_count >= 1
    assert "не найдены" in response.answer.lower()
    assert "LLM отключён" not in response.answer


@pytest.mark.asyncio
async def test_react_empty_context_returns_no_sources() -> None:
    settings = Settings(_env_file=None, llm_enabled=True, agent_relevance_filter_enabled=False)
    llm = _ReactSequenceLLM([_react_action("Done", "finish")], "unused")
    agent, _ = _agent(results=[], llm=llm, literature=_FakeLiteratureService(papers=[]), settings=settings)

    response = await agent.run(AgentAskRequest(message="question?", mode="question"))

    assert "не найдены" in response.answer.lower()
    assert "llm отключён" not in response.answer.lower()


@pytest.mark.asyncio
async def test_react_bootstrap_gives_context_when_llm_finishes_immediately() -> None:
    llm = _ReactSequenceLLM([_react_action("Done", "finish")], _LONG_ANSWER)
    agent, literature = _agent(
        results=[_result(0.9, text="Important local fact about debt.")],
        llm=llm,
    )

    response = await agent.run(AgentAskRequest(message="question?", mode="question"))

    assert literature.call_count == 0
    assert "local_hybrid_search" in [step.tool for step in response.steps]
    assert response.sources.local
    assert "LLM отключён" not in response.answer


@pytest.mark.asyncio
async def test_react_llm_disabled_empty_context_message() -> None:
    agent, _ = _agent(results=[], llm=None, literature=_FakeLiteratureService(papers=[]))

    response = await agent.run(AgentAskRequest(message="question?", mode="question"))

    assert "LLM отключён" in response.answer
    assert "не найдены" in response.answer.lower()
