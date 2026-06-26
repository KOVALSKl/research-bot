from __future__ import annotations

from research_shared.config.settings import Settings
from research_shared.domain.models import SearchResult
from research_shared.llm.protocols import LLMProvider
from research_shared.logging_config import get_logger

logger = get_logger(__name__)

_CONTEXT_CHECK_QUESTION = (
    "Достаточно ли **релевантного** предоставленного контекста для полного ответа "
    "на вопрос? Ответь только yes или no."
)


def _rule_based_sufficient(results: list[SearchResult], settings: Settings) -> bool:
    if not results:
        return False
    if len(results) < settings.min_local_results:
        return False
    top_score = max(result.score for result in results)
    return top_score >= settings.min_local_score


def _parse_yes_no(answer: str) -> bool | None:
    normalized = answer.strip().lower()
    if normalized.startswith("yes") or normalized.startswith("да"):
        return True
    if normalized.startswith("no") or normalized.startswith("нет"):
        return False
    return None


def _llm_context_sufficient(
    llm: LLMProvider,
    query: str,
    context: str,
) -> bool | None:
    truncated = context[:3000] if context else "No context available."
    answer = llm.generate(
        f"{_CONTEXT_CHECK_QUESTION}\n\nВопрос: {query}",
        truncated,
    )
    parsed = _parse_yes_no(answer)
    if parsed is None:
        logger.warning(
            "LLM context check returned unparseable answer",
            extra={"event": "agent.context_check.unparseable", "answer": answer[:100]},
        )
    return parsed


def is_context_sufficient(
    results: list[SearchResult],
    settings: Settings,
    *,
    query: str = "",
    context: str = "",
    llm: LLMProvider | None = None,
) -> bool:
    rule_based = _rule_based_sufficient(results, settings)

    if not settings.agent_use_llm_context_check or llm is None:
        return rule_based

    try:
        llm_result = _llm_context_sufficient(llm, query, context)
    except Exception:
        logger.exception(
            "LLM context check failed; falling back to rule-based",
            extra={"event": "agent.context_check.llm_error"},
        )
        return rule_based

    if llm_result is None:
        return rule_based
    return llm_result
