from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, SparseVector

from research_shared.config.settings import Settings
from research_shared.domain.models import ResearchChunk
from research_shared.storage.protocols import DenseEmbedder, SparseEncoder
from research_shared.storage.qdrant.collection import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME


def _chunk_payload(chunk: ResearchChunk) -> dict:
    page = chunk.metadata.get("page")
    chapter = chunk.chapter or chunk.metadata.get("chapter")
    metadata = {
        key: value
        for key, value in chunk.metadata.items()
        if key not in ("page", "chapter")
    }

    return {
        "research_id": chunk.research_id,
        "research_title": chunk.title,
        "text": chunk.text,
        "source_path": chunk.source_path or "",
        "display_name": chunk.display_name or "",
        "authors": chunk.authors or [],
        "page": page,
        "chapter": chapter,
        "metadata": metadata,
    }


class QdrantVectorStore:
    """Upsert and delete research chunks in Qdrant."""

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

    async def upsert(self, chunks: list[ResearchChunk]) -> int:
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        dense_vectors = self._dense_embedder.embed(texts)
        sparse_vectors = self._sparse_encoder.encode(texts)

        points: list[PointStruct] = []
        for chunk, dense, (indices, values) in zip(chunks, dense_vectors, sparse_vectors, strict=True):
            points.append(
                PointStruct(
                    id=chunk.id,
                    vector={
                        DENSE_VECTOR_NAME: dense,
                        SPARSE_VECTOR_NAME: SparseVector(indices=indices, values=values),
                    },
                    payload=_chunk_payload(chunk),
                )
            )

        await self._client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    async def delete_by_ids(self, ids: list[str]) -> int:
        if not ids:
            return 0
        await self._client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )
        return len(ids)

    async def delete_by_research_id(self, research_id: str) -> int:
        filter_ = Filter(must=[FieldCondition(key="research_id", match=MatchValue(value=research_id))])
        await self._client.delete(
            collection_name=self.collection_name,
            points_selector=filter_,
        )
        return 1
