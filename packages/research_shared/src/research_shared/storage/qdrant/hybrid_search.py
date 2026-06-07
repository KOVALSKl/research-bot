from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from research_shared.config.settings import Settings
from research_shared.domain.models import ResearchChunk, SearchQuery, SearchResult, SearchType
from research_shared.storage.protocols import DenseEmbedder, HybridSearcher, SparseEncoder
from research_shared.storage.qdrant.collection import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME


class QdrantHybridSearchService:
    """Hybrid search combining dense semantic and sparse term-based retrieval."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        dense_embedder: DenseEmbedder,
        sparse_encoder: SparseEncoder,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._dense_embedder = dense_embedder
        self._sparse_encoder = sparse_encoder
        self._settings = settings or Settings()

    @property
    def collection_name(self) -> str:
        return self._settings.qdrant_collection_name

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        if query.search_type == SearchType.DENSE:
            return await self._search_dense(query)
        if query.search_type == SearchType.SPARSE:
            return await self._search_sparse(query)
        return await self._search_hybrid(query)

    async def _search_hybrid(self, query: SearchQuery) -> list[SearchResult]:
        dense_vector = self._dense_embedder.embed([query.query])[0]
        sparse_indices, sparse_values = self._sparse_encoder.encode([query.query])[0]

        prefetch_limit = max(query.limit * 2, 20)
        response = await self._client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=query.limit,
            score_threshold=query.score_threshold,
            with_payload=True,
        )

        return self._points_to_results(response.points, SearchType.HYBRID)

    async def _search_dense(self, query: SearchQuery) -> list[SearchResult]:
        dense_vector = self._dense_embedder.embed([query.query])[0]
        response = await self._client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using=DENSE_VECTOR_NAME,
            limit=query.limit,
            score_threshold=query.score_threshold,
            with_payload=True,
        )
        return self._points_to_results(response.points, SearchType.DENSE)

    async def _search_sparse(self, query: SearchQuery) -> list[SearchResult]:
        sparse_indices, sparse_values = self._sparse_encoder.encode([query.query])[0]
        response = await self._client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(indices=sparse_indices, values=sparse_values),
            using=SPARSE_VECTOR_NAME,
            limit=query.limit,
            score_threshold=query.score_threshold,
            with_payload=True,
        )
        return self._points_to_results(response.points, SearchType.SPARSE)

    async def search_by_research_id(
        self,
        query: SearchQuery,
        research_id: str,
    ) -> list[SearchResult]:
        """Hybrid search scoped to a single research document."""
        dense_vector = self._dense_embedder.embed([query.query])[0]
        sparse_indices, sparse_values = self._sparse_encoder.encode([query.query])[0]
        research_filter = Filter(
            must=[FieldCondition(key="research_id", match=MatchValue(value=research_id))]
        )

        prefetch_limit = max(query.limit * 2, 20)
        response = await self._client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=research_filter,
                ),
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=research_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=query.limit,
            score_threshold=query.score_threshold,
            with_payload=True,
        )
        return self._points_to_results(response.points, SearchType.HYBRID)

    @staticmethod
    def _points_to_results(points, search_type: SearchType) -> list[SearchResult]:
        results: list[SearchResult] = []
        for point in points:
            payload = point.payload or {}
            chunk = ResearchChunk(
                id=str(point.id),
                research_id=str(payload.get("research_id", "")),
                title=str(payload.get("title", "")),
                text=str(payload.get("text", "")),
                metadata=payload.get("metadata") or {},
            )
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=point.score or 0.0,
                    search_type=search_type,
                )
            )
        return results


# HybridSearcher protocol conformance check at import is skipped — async methods
