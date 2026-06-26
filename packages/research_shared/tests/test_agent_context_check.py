import pytest

from research_shared.agents.context_check import is_context_sufficient
from research_shared.config.settings import Settings
from research_shared.domain.models import ResearchChunk, SearchResult, SearchType


class _FakeLLM:
    def __init__(self, answer: str = "yes") -> None:
        self.answer = answer
        self.calls = 0

    def generate(self, question: str, context: str) -> str:
        self.calls += 1
        return self.answer


def _result(score: float) -> SearchResult:
    return SearchResult(
        chunk=ResearchChunk(research_id="r1", title="T", text="body"),
        score=score,
        search_type=SearchType.HYBRID,
    )


def test_empty_results_insufficient() -> None:
    settings = Settings(_env_file=None)
    assert is_context_sufficient([], settings) is False


def test_below_min_results_insufficient() -> None:
    settings = Settings(_env_file=None, min_local_results=3, min_local_score=0.5)
    results = [_result(0.9), _result(0.8)]
    assert is_context_sufficient(results, settings) is False


def test_low_top_score_insufficient() -> None:
    settings = Settings(_env_file=None, min_local_results=2, min_local_score=0.5)
    results = [_result(0.4), _result(0.3)]
    assert is_context_sufficient(results, settings) is False


def test_threshold_values_sufficient() -> None:
    settings = Settings(_env_file=None, min_local_results=3, min_local_score=0.5)
    results = [_result(0.5), _result(0.5), _result(0.5)]
    assert is_context_sufficient(results, settings) is True


def test_llm_no_overrides_rule_based_when_flag_false() -> None:
    settings = Settings(_env_file=None, agent_use_llm_context_check=False)
    llm = _FakeLLM(answer="no")
    results = [_result(0.9), _result(0.8), _result(0.7)]

    assert is_context_sufficient(results, settings, llm=llm) is True
    assert llm.calls == 0


def test_llm_no_when_flag_true() -> None:
    settings = Settings(_env_file=None, agent_use_llm_context_check=True)
    llm = _FakeLLM(answer="no")
    results = [_result(0.9), _result(0.8), _result(0.7)]

    assert is_context_sufficient(
        results,
        settings,
        query="question",
        context="[1] context",
        llm=llm,
    ) is False
    assert llm.calls == 1


def test_llm_failure_falls_back_to_rule_based() -> None:
    settings = Settings(_env_file=None, agent_use_llm_context_check=True)

    class _BrokenLLM:
        def generate(self, question: str, context: str) -> str:
            raise RuntimeError("llm down")

    results = [_result(0.9), _result(0.8), _result(0.7)]
    assert is_context_sufficient(
        results,
        settings,
        query="q",
        context="ctx",
        llm=_BrokenLLM(),
    ) is True


def test_filtered_empty_always_insufficient() -> None:
    settings = Settings(_env_file=None, min_local_results=1, min_local_score=0.1)
    assert is_context_sufficient([], settings) is False


def test_llm_asks_about_relevant_context() -> None:
    settings = Settings(_env_file=None, agent_use_llm_context_check=True)
    llm = _FakeLLM(answer="yes")
    results = [_result(0.9)]

    is_context_sufficient(
        results,
        settings,
        query="question",
        context="[1] relevant",
        llm=llm,
    )

    assert llm.calls == 1
