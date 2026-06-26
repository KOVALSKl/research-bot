from __future__ import annotations

from dataclasses import dataclass

from research_shared.agents.models import AgentStep, ResolvedAgentMode
from research_shared.config.settings import Settings, get_settings


@dataclass(frozen=True)
class SearchAttestation:
    bootstrap_ran: bool
    react_searches: int
    safety_net_ran: bool
    local_chunks: int
    external_papers: int

    @property
    def search_had_results(self) -> bool:
        return self.local_chunks > 0 or self.external_papers > 0

    def format_detail(self) -> str:
        return (
            f"bootstrap={self.bootstrap_ran};"
            f"react_searches={self.react_searches};"
            f"safety_net={self.safety_net_ran};"
            f"local={self.local_chunks};"
            f"external={self.external_papers}"
        )


def build_search_attestation(
    steps: list[AgentStep],
    *,
    local_chunks: int,
    external_papers: int,
) -> SearchAttestation:
    bootstrap_ran = False
    safety_net_ran = False
    react_searches = 0

    for step in steps:
        detail = step.detail or ""
        if step.tool == "local_hybrid_search":
            if "bootstrap" in detail:
                bootstrap_ran = True
            elif "safety_net" in detail:
                safety_net_ran = True
            elif "mandatory" not in detail:
                react_searches += 1
        elif step.tool == "external_literature_search":
            if "safety_net" in detail:
                safety_net_ran = True
            elif "mandatory" not in detail:
                react_searches += 1

    return SearchAttestation(
        bootstrap_ran=bootstrap_ran,
        react_searches=react_searches,
        safety_net_ran=safety_net_ran,
        local_chunks=local_chunks,
        external_papers=external_papers,
    )


def search_sufficient(
    attestation: SearchAttestation,
    mode: ResolvedAgentMode,
    settings: Settings | None = None,
) -> bool:
    if mode == "idea_evaluation":
        cfg = settings or get_settings()
        strong_local = attestation.local_chunks >= cfg.agent_idea_min_local_chunks
        return (
            attestation.external_papers >= cfg.agent_idea_min_external_papers
            or strong_local
        )
    return attestation.search_had_results


def build_empty_context_message(
    *,
    mode: ResolvedAgentMode,
    llm_enabled: bool,
    llm_available: bool,
    attestation: SearchAttestation,
) -> str:
    if not llm_enabled:
        if mode == "question":
            return "LLM отключён. Релевантные источники не найдены."
        return "LLM отключён. Для оценки идеи нужен включённый LLM и найденные источники."

    if not llm_available:
        if mode == "question":
            return (
                "LLM недоступен (проверьте Ollama или Hugging Face). "
                "Источники в PDF и внешних базах не найдены. "
                "Попробуйте переформулировать вопрос или загрузить PDF."
            )
        return (
            "LLM недоступен (проверьте Ollama или Hugging Face). "
            "Для оценки идеи не найдены релевантные источники."
        )

    if mode == "question":
        return (
            "Релевантные источники не найдены в загруженных PDF и внешних базах. "
            "Попробуйте переформулировать вопрос или загрузить PDF-документы в бот."
        )
    detail = attestation.format_detail()
    if attestation.external_papers == 0 and attestation.local_chunks == 0:
        return (
            "Для оценки идеи не найдены релевантные источники. "
            "Загрузите PDF или уточните формулировку идеи. "
            f"({detail})"
        )
    return (
        "Недостаточно источников для полноценной оценки идеи. "
        f"({detail})"
    )
