from __future__ import annotations

import re
from dataclasses import dataclass

from research_shared.agents.tools.local_search import LocalSearchResult
from research_shared.config.settings import Settings
from research_shared.domain.models import Citation, SearchResult
from research_shared.llm.prompts import get_agent_relevance_prompt
from research_shared.llm.protocols import LLMProvider
from research_shared.logging_config import get_logger
from research_shared.rag.citations import dedupe_citations
from research_shared.rag.service import RagService

logger = get_logger(__name__)


@dataclass(frozen=True)
class FilteredLocalResult:
    results: list[SearchResult]
    citations: list[Citation]
    context: str
    filtered_count: int


def _rebuild_local_result(results: list[SearchResult]) -> tuple[list[Citation], str]:
    raw_citations = [
        Citation(
            research_id=result.chunk.research_id,
            title=result.chunk.title,
            page=result.chunk.metadata.get("page"),
            score=result.score,
            source_path=result.chunk.source_path,
            display_name=result.chunk.display_name,
            chapter=result.chunk.chapter,
            authors=result.chunk.authors,
        )
        for result in results
    ]
    citations = dedupe_citations(raw_citations)
    context = RagService._build_context(results, citations)
    return citations, context


def _apply_l1_filter(
    results: list[SearchResult],
    settings: Settings,
) -> list[SearchResult]:
    threshold = settings.agent_min_chunk_score
    return [result for result in results if result.score >= threshold]


def _parse_relevant_indices(answer: str, max_index: int) -> list[int] | None:
    normalized = answer.strip().lower()
    if normalized in {"none", "нет", "no"}:
        return []
    numbers = re.findall(r"\d+", normalized)
    if not numbers:
        return None
    indices = sorted({int(value) for value in numbers if 1 <= int(value) <= max_index})
    return indices


def _llm_filter_indices(
    llm: LLMProvider,
    question: str,
    context: str,
    settings: Settings,
    max_index: int,
) -> list[int] | None:
    system_prompt = get_agent_relevance_prompt(settings)
    if hasattr(llm, "_system_prompt"):
        original = llm._system_prompt
        llm._system_prompt = system_prompt
        try:
            answer = llm.generate(question, context)
        finally:
            llm._system_prompt = original
    else:
        answer = llm.generate(question, context)
    return _parse_relevant_indices(answer, max_index)


async def filter_relevant_results(
    question: str,
    local: LocalSearchResult,
    settings: Settings,
    llm: LLMProvider | None = None,
) -> FilteredLocalResult:
    original_count = len(local.results)
    if not settings.agent_relevance_filter_enabled or not local.results:
        return FilteredLocalResult(
            results=local.results,
            citations=local.citations,
            context=local.context,
            filtered_count=0,
        )

    l1_results = _apply_l1_filter(local.results, settings)
    l1_removed = original_count - len(l1_results)

    if not l1_results:
        citations, context = _rebuild_local_result([])
        return FilteredLocalResult(
            results=[],
            citations=citations,
            context=context,
            filtered_count=l1_removed,
        )

    l1_citations, l1_context = _rebuild_local_result(l1_results)

    if not settings.effective_agent_use_llm_relevance_filter() or llm is None:
        return FilteredLocalResult(
            results=l1_results,
            citations=l1_citations,
            context=l1_context,
            filtered_count=l1_removed,
        )

    try:
        indices = _llm_filter_indices(
            llm,
            question,
            l1_context,
            settings,
            max_index=len(l1_results),
        )
    except Exception:
        logger.exception(
            "LLM relevance filter failed; falling back to L1-only",
            extra={"event": "agent.relevance_filter.llm_error"},
        )
        return FilteredLocalResult(
            results=l1_results,
            citations=l1_citations,
            context=l1_context,
            filtered_count=l1_removed,
        )

    if indices is None:
        logger.warning(
            "LLM relevance filter returned unparseable answer; falling back to L1-only",
            extra={"event": "agent.relevance_filter.unparseable"},
        )
        return FilteredLocalResult(
            results=l1_results,
            citations=l1_citations,
            context=l1_context,
            filtered_count=l1_removed,
        )

    if not indices:
        citations, context = _rebuild_local_result([])
        return FilteredLocalResult(
            results=[],
            citations=citations,
            context=context,
            filtered_count=original_count,
        )

    filtered_results = [
        l1_results[index - 1] for index in indices if 1 <= index <= len(l1_results)
    ]
    citations, context = _rebuild_local_result(filtered_results)
    filtered_count = original_count - len(filtered_results)
    return FilteredLocalResult(
        results=filtered_results,
        citations=citations,
        context=context,
        filtered_count=filtered_count,
    )
