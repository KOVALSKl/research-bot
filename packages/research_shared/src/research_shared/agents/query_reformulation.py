from __future__ import annotations

from research_shared.config.settings import Settings
from research_shared.llm.prompts import (
    get_agent_idea_query_reformulation_prompt,
    get_agent_query_reformulation_prompt,
)
from research_shared.llm.protocols import LLMProvider
from research_shared.logging_config import get_logger

logger = get_logger(__name__)


def _dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        normalized = query.strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def _parse_reformulation_lines(raw: str, original: str, max_queries: int) -> list[str]:
    lines = [line.strip().lstrip("-•0123456789.) ") for line in raw.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return [original]
    queries = _dedupe_queries([original, *lines])
    return queries[:max_queries]


async def _generate_reformulation_with_prompt(
    question: str,
    llm: LLMProvider,
    system_prompt: str,
) -> str:
    if hasattr(llm, "_system_prompt"):
        original = llm._system_prompt
        llm._system_prompt = system_prompt
        try:
            return llm.generate(question, "")
        finally:
            llm._system_prompt = original
    return llm.generate(question, "")


async def _generate_reformulation(
    question: str,
    llm: LLMProvider,
    settings: Settings,
) -> str:
    system_prompt = get_agent_query_reformulation_prompt(settings)
    return await _generate_reformulation_with_prompt(question, llm, system_prompt)


def _question_aspect_fallback(question: str, max_queries: int) -> list[str]:
    """Fallback queries when LLM is unavailable: original + keyword variants for external search."""
    base = question.strip()
    aspects = [
        base,
        f"{base} method approach",
        f"{base} research",
    ]
    return _dedupe_queries(aspects)[:max_queries]


async def build_search_queries(
    question: str,
    llm: LLMProvider | None,
    settings: Settings,
) -> list[str]:
    original = question.strip()
    if not original:
        return []

    if not settings.agent_query_reformulation_enabled or llm is None:
        return _question_aspect_fallback(original, settings.agent_search_queries_max)

    try:
        raw = await _generate_reformulation(original, llm, settings)
    except Exception:
        logger.exception(
            "Query reformulation failed; using original question",
            extra={"event": "agent.query_reformulation.error"},
        )
        return [original]

    return _parse_reformulation_lines(
        raw.strip(),
        original,
        settings.agent_search_queries_max,
    )


def _idea_aspect_fallback(idea: str, max_queries: int) -> list[str]:
    base = idea.strip()
    aspects = [
        base,
        f"{base} метод подход",
        f"{base} research problem domain",
    ]
    return _dedupe_queries(aspects)[:max_queries]


async def build_idea_search_queries(
    idea: str,
    llm: LLMProvider | None,
    settings: Settings,
) -> list[str]:
    original = idea.strip()
    if not original:
        return []

    if not settings.agent_idea_query_reformulation_enabled or llm is None:
        return _idea_aspect_fallback(original, settings.agent_search_queries_max)

    system_prompt = get_agent_idea_query_reformulation_prompt(settings)
    try:
        raw = await _generate_reformulation_with_prompt(original, llm, system_prompt)
    except Exception:
        logger.exception(
            "Idea query reformulation failed; using original idea text",
            extra={"event": "agent.idea_query_reformulation.error"},
        )
        return _idea_aspect_fallback(original, settings.agent_search_queries_max)

    queries = _parse_reformulation_lines(
        raw.strip(),
        original,
        settings.agent_search_queries_max,
    )
    if len(queries) < 2:
        return _idea_aspect_fallback(original, settings.agent_search_queries_max)
    return queries


async def reformulate_question(
    question: str,
    llm: LLMProvider | None,
    settings: Settings,
) -> str:
    queries = await build_search_queries(question, llm, settings)
    if not queries:
        return question.strip()
    return queries[-1] if len(queries) > 1 else queries[0]
