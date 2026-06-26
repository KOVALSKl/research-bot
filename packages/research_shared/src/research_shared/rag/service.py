from __future__ import annotations

import re
from typing import TYPE_CHECKING

from research_shared.domain.models import (
    AskQuery,
    AskResponse,
    Citation,
    SearchQuery,
    SearchResult,
    SearchType,
    SourceFileRef,
)
from research_shared.logging_config import get_logger
from research_shared.rag.citations import (
    build_source_files,
    citation_display_name,
    citation_filename,
    dedupe_citations,
)
from research_shared.storage.protocols import HybridSearcher

if TYPE_CHECKING:
    from research_shared.llm.protocols import LLMProvider

logger = get_logger(__name__)


class RagService:
    """Assembles a RAG answer: hybrid search → context + citations → (opt.) LLM.

    Reuses the existing v1.1 hybrid search. The LLM provider is optional; when
    absent, the response carries only ``citations`` and ``context_chunks`` with
    ``answer=None``.
    """

    def __init__(
        self,
        searcher: HybridSearcher,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self._searcher = searcher
        self._llm = llm_provider

    async def ask(self, query: AskQuery) -> AskResponse:
        results = await self._searcher.search(
            SearchQuery(
                query=query.question,
                limit=query.limit,
                search_type=SearchType.HYBRID,
            )
        )

        context_chunks = [result.chunk for result in results]
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
                source_url=result.chunk.metadata.get("source_url"),
            )
            for result in results
        ]
        citations = dedupe_citations(raw_citations)
        source_files = build_source_files(citations)

        answer: str | None = None
        if self._llm is not None:
            raw_answer = self._llm.generate(
                query.question,
                self._build_context(results, citations),
            )
            answer = self._dedupe_answer(raw_answer)

        return AskResponse(
            answer=answer,
            citations=citations,
            context_chunks=context_chunks,
            source_files=source_files,
        )

    @staticmethod
    def _build_source_files(citations: list[Citation]) -> list[SourceFileRef]:
        return build_source_files(citations)

    @staticmethod
    def _build_context(
        results: list[SearchResult],
        citations: list[Citation],
    ) -> str:
        chunk_by_key: dict[tuple[str, int | None, str], SearchResult] = {}
        for result in results:
            citation = Citation(
                research_id=result.chunk.research_id,
                title=result.chunk.title,
                page=result.chunk.metadata.get("page"),
                score=result.score,
                source_path=result.chunk.source_path,
                display_name=result.chunk.display_name,
                authors=result.chunk.authors,
                chapter=result.chunk.chapter,
            )
            key = (
                citation.research_id,
                citation.page,
                citation_display_name(citation),
            )
            existing = chunk_by_key.get(key)
            if existing is None or result.score > existing.score:
                chunk_by_key[key] = result

        parts: list[str] = []
        for index, citation in enumerate(citations, start=1):
            key = (
                citation.research_id,
                citation.page,
                citation_display_name(citation),
            )
            result = chunk_by_key.get(key)
            if result is None:
                continue
            page = result.chunk.metadata.get("page")
            location_parts: list[str] = []
            if page is not None:
                location_parts.append(f"p. {page}")
            if result.chunk.chapter:
                location_parts.append(f'гл. «{result.chunk.chapter}»')
            location = f", {', '.join(location_parts)}" if location_parts else ""
            parts.append(
                f"[{index}] {result.chunk.title}{location}\n{result.chunk.text}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _dedupe_answer(answer: str) -> str:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", answer) if part.strip()]
        if len(paragraphs) < 2:
            return answer

        unique: list[str] = []
        seen: set[str] = set()
        removed = False
        for paragraph in paragraphs:
            normalized = " ".join(paragraph.split())
            if normalized in seen:
                removed = True
                continue
            seen.add(normalized)
            unique.append(paragraph)

        if not removed:
            return answer

        logger.info(
            "LLM answer deduplicated",
            extra={"event": "ask.answer_deduped", "count": len(paragraphs) - len(unique)},
        )
        return "\n\n".join(unique)
