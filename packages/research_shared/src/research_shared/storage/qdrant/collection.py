from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from research_shared.config.settings import Settings
from research_shared.storage.embeddings.factory import probe_dense_vector_size

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


async def ensure_collection(
    client: AsyncQdrantClient,
    settings: Settings | None = None,
    vector_size: int | None = None,
) -> None:
    settings = settings or Settings()
    collection_name = settings.qdrant_collection_name
    expected_size = vector_size or probe_dense_vector_size(settings)

    if await client.collection_exists(collection_name):
        current_size = await _get_dense_vector_size(client, collection_name)
        if current_size is not None and current_size != expected_size:
            if not settings.qdrant_recreate_on_schema_mismatch:
                raise RuntimeError(
                    f"Collection '{collection_name}' dense vector size is {current_size}, "
                    f"expected {expected_size}. Set QDRANT_RECREATE_ON_SCHEMA_MISMATCH=true "
                    "or delete the collection manually and re-index."
                )
            await client.delete_collection(collection_name)
        else:
            return

    await client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=expected_size, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )

    await client.create_payload_index(
        collection_name=collection_name,
        field_name="research_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )


async def _get_dense_vector_size(client: AsyncQdrantClient, collection_name: str) -> int | None:
    info = await client.get_collection(collection_name)
    vectors = info.config.params.vectors
    if vectors is None:
        return None
    if isinstance(vectors, dict):
        dense = vectors.get(DENSE_VECTOR_NAME)
        return dense.size if dense is not None else None
    return vectors.size
