from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Prefetch,
    Range,
    SparseVector,
)

from research_shared.config.settings import Settings
from research_shared.domain.models import (
    ResearchChunk,
    SearchFilters,
    SearchQuery,
    SearchResult,
    SearchType,
)
from research_shared.storage.protocols import DenseEmbedder, HybridSearcher, SparseEncoder
from research_shared.storage.qdrant.collection import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME


def _build_filter(filters: SearchFilters | None) -> Filter | None:
    if filters is None:
        return None

    conditions: list[FieldCondition] = []

    if filters.research_id is not None:
        conditions.append(
            FieldCondition(key="research_id", match=MatchValue(value=filters.research_id))
        )
    if filters.source_path is not None:
        conditions.append(
            FieldCondition(key="source_path", match=MatchValue(value=filters.source_path))
        )
    if filters.authors:
        conditions.append(FieldCondition(key="authors", match=MatchAny(any=filters.authors)))
    if filters.chapter is not None:
        conditions.append(
            FieldCondition(key="chapter", match=MatchValue(value=filters.chapter))
        )

    page_range: dict[str, int] = {}
    if filters.page_min is not None:
        page_range["gte"] = filters.page_min
    if filters.page_max is not None:
        page_range["lte"] = filters.page_max
    if page_range:
        conditions.append(FieldCondition(key="page", range=Range(**page_range)))

    if not conditions:
        return None
    return Filter(must=conditions)


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
        search_filter = _build_filter(query.filters)

        prefetch_limit = max(query.limit * 2, 20)
        response = await self._client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=search_filter,
                ),
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=search_filter,
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
            query_filter=_build_filter(query.filters),
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
            query_filter=_build_filter(query.filters),
        )
        return self._points_to_results(response.points, SearchType.SPARSE)

    async def search_by_research_id(
        self,
        query: SearchQuery,
        research_id: str,
    ) -> list[SearchResult]:
        """Hybrid search scoped to a single research document."""
        base_filters = query.filters or SearchFilters()
        scoped_filters = base_filters.model_copy(update={"research_id": research_id})
        scoped_query = query.model_copy(update={"filters": scoped_filters})
        return await self.search(scoped_query)

    @staticmethod
    def _points_to_results(points, search_type: SearchType) -> list[SearchResult]:
        results: list[SearchResult] = []
        for point in points:
            payload = point.payload or {}
            metadata = dict(payload.get("metadata") or {})

            research_title = payload.get("research_title") or payload.get("title", "")
            page = payload.get("page")
            if page is None:
                page = metadata.get("page")

            source_path = payload.get("source_path") or ""
            display_name = payload.get("display_name") or None
            if display_name == "":
                display_name = None
            authors = payload.get("authors") or []
            if not isinstance(authors, list):
                authors = []

            chapter = payload.get("chapter")
            if chapter is None:
                chapter = metadata.get("chapter")

            if page is not None and "page" not in metadata:
                metadata["page"] = page
            if chapter is not None and "chapter" not in metadata:
                metadata["chapter"] = chapter

            chunk = ResearchChunk(
                id=str(point.id),
                research_id=str(payload.get("research_id", "")),
                title=str(research_title),
                text=str(payload.get("text", "")),
                source_path=source_path or None,
                display_name=display_name,
                authors=authors,
                chapter=chapter,
                metadata=metadata,
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
