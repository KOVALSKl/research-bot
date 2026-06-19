import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from research_shared.config.settings import Settings
from research_shared.domain.models import DocumentRecord

_NAMESPACE = uuid.UUID("6f1d8c2e-2a4b-5c6d-8e9f-000000000002")
_PLACEHOLDER_VECTOR = [0.0]


def _point_id(filename: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, filename))


class QdrantIngestionStateStore:
    """Durable per-file ingestion state in a dedicated Qdrant collection.

    One point per source file (id = UUID5(filename)), placeholder vector of
    dim 1, payload mirrors :class:`DocumentRecord`. Safe across workers via
    atomic upserts (no file locks).
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._settings = settings or Settings()

    @property
    def collection_name(self) -> str:
        return self._settings.ingest_state_collection

    async def ensure_collection(self) -> None:
        if await self._client.collection_exists(self.collection_name):
            return
        await self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )

    async def get(self, filename: str) -> DocumentRecord | None:
        points = await self._client.retrieve(
            collection_name=self.collection_name,
            ids=[_point_id(filename)],
            with_payload=True,
        )
        if not points:
            return None
        payload = points[0].payload or {}
        return DocumentRecord(**payload)

    async def upsert(self, record: DocumentRecord) -> None:
        await self._client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=_point_id(record.filename),
                    vector=_PLACEHOLDER_VECTOR,
                    payload=record.model_dump(mode="json"),
                )
            ],
        )

    async def list(self) -> list[DocumentRecord]:
        records: list[DocumentRecord] = []
        offset = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for point in points:
                payload = point.payload or {}
                records.append(DocumentRecord(**payload))
            if offset is None:
                break
        return records
