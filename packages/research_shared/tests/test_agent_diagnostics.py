from research_shared.agents.diagnostics import (
    SearchAttestation,
    build_empty_context_message,
    build_search_attestation,
    search_sufficient,
)
from research_shared.agents.models import AgentStep


def test_build_search_attestation_detects_bootstrap_and_safety_net() -> None:
    steps = [
        AgentStep(
            tool="local_hybrid_search",
            query="q",
            results_count=0,
            detail="bootstrap; 0→0 chunks",
        ),
        AgentStep(
            tool="external_literature_search",
            query="q",
            results_count=2,
            detail="safety_net; openalex:2",
        ),
    ]
    attestation = build_search_attestation(steps, local_chunks=0, external_papers=2)
    assert attestation.bootstrap_ran is True
    assert attestation.safety_net_ran is True
    assert attestation.external_papers == 2
    assert attestation.search_had_results is True


def test_search_sufficient_requires_results() -> None:
    empty = SearchAttestation(
        bootstrap_ran=True,
        react_searches=0,
        safety_net_ran=True,
        local_chunks=0,
        external_papers=0,
    )
    assert search_sufficient(empty, "question") is False


def test_search_sufficient_idea_requires_external_or_strong_local() -> None:
    from research_shared.config.settings import Settings

    settings = Settings(_env_file=None, agent_idea_min_external_papers=1, agent_idea_min_local_chunks=2)
    weak = SearchAttestation(
        bootstrap_ran=True,
        react_searches=1,
        safety_net_ran=False,
        local_chunks=1,
        external_papers=0,
    )
    assert search_sufficient(weak, "idea_evaluation", settings) is False

    with_external = SearchAttestation(
        bootstrap_ran=True,
        react_searches=1,
        safety_net_ran=False,
        local_chunks=0,
        external_papers=2,
    )
    assert search_sufficient(with_external, "idea_evaluation", settings) is True


def test_empty_message_llm_enabled_without_llm_available() -> None:
    attestation = SearchAttestation(
        bootstrap_ran=True,
        react_searches=1,
        safety_net_ran=True,
        local_chunks=0,
        external_papers=0,
    )
    message = build_empty_context_message(
        mode="question",
        llm_enabled=True,
        llm_available=False,
        attestation=attestation,
    )
    assert "LLM недоступен" in message
    assert "LLM отключён" not in message


def test_empty_message_llm_enabled_with_no_sources() -> None:
    attestation = SearchAttestation(
        bootstrap_ran=True,
        react_searches=1,
        safety_net_ran=True,
        local_chunks=0,
        external_papers=0,
    )
    message = build_empty_context_message(
        mode="question",
        llm_enabled=True,
        llm_available=True,
        attestation=attestation,
    )
    assert "не найдены" in message.lower()
    assert "LLM отключён" not in message
