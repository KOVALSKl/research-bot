from typing import Protocol, runtime_checkable

from research_shared.domain.models import ResearchChunk, SearchQuery, SearchResult


@runtime_checkable
class DenseEmbedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class SparseEncoder(Protocol):
    def encode(self, texts: list[str]) -> list[tuple[list[int], list[float]]]: ...


@runtime_checkable
class VectorStore(Protocol):
    async def upsert(self, chunks: list[ResearchChunk]) -> int: ...

    async def delete_by_ids(self, ids: list[str]) -> int: ...

    async def delete_by_research_id(self, research_id: str) -> int: ...


@runtime_checkable
class HybridSearcher(Protocol):
    async def search(self, query: SearchQuery) -> list[SearchResult]: ...
