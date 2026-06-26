from dataclasses import dataclass

from research_shared.domain.models import Citation, SearchQuery, SearchResult, SearchType
from research_shared.rag.citations import dedupe_citations
from research_shared.rag.service import RagService
from research_shared.storage.protocols import HybridSearcher


@dataclass(frozen=True)
class LocalSearchResult:
    results: list[SearchResult]
    citations: list[Citation]
    context: str


def _chunk_dedupe_key(result: SearchResult) -> str:
    chunk = result.chunk
    page = chunk.metadata.get("page")
    return f"rp:{chunk.research_id}:{page}:{chunk.text[:80]}"


def merge_local_search_results(parts: list[LocalSearchResult]) -> LocalSearchResult:
    if not parts:
        return LocalSearchResult(results=[], citations=[], context="")
    if len(parts) == 1:
        return parts[0]

    seen_keys: set[str] = set()
    merged_results: list[SearchResult] = []
    for part in parts:
        for result in part.results:
            key = _chunk_dedupe_key(result)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_results.append(result)

    merged_results.sort(key=lambda item: item.score, reverse=True)

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
        for result in merged_results
    ]
    citations = dedupe_citations(raw_citations)
    context = RagService._build_context(merged_results, citations)

    return LocalSearchResult(
        results=merged_results,
        citations=citations,
        context=context,
    )


async def local_hybrid_search(
    searcher: HybridSearcher,
    rag_service: RagService,
    query: str,
    limit: int,
) -> LocalSearchResult:
    results = await searcher.search(
        SearchQuery(
            query=query,
            limit=limit,
            search_type=SearchType.HYBRID,
        )
    )

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
    context = RagService._build_context(results, citations)

    return LocalSearchResult(
        results=results,
        citations=citations,
        context=context,
    )
