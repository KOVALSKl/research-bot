from unittest.mock import AsyncMock, MagicMock

import pytest

from research_shared.domain.models import ResearchChunk, SearchQuery, SearchType
from research_shared.storage.qdrant.hybrid_search import QdrantHybridSearchService
from research_shared.storage.qdrant.store import QdrantVectorStore


class MockDenseEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class MockSparseEncoder:
    def encode(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return [([1, 2], [0.5, 0.3]) for _ in texts]


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.upsert = AsyncMock()
    client.delete = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_vector_store_upsert(mock_client: AsyncMock) -> None:
    store = QdrantVectorStore(
        mock_client,
        MockDenseEmbedder(),
        MockSparseEncoder(),
    )
    chunks = [
        ResearchChunk(
            id="c1",
            research_id="r1",
            title="Title",
            text="Some research text.",
        )
    ]
    count = await store.upsert(chunks)
    assert count == 1
    mock_client.upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_hybrid_search_returns_results(mock_client: AsyncMock) -> None:
    point = MagicMock()
    point.id = "c1"
    point.score = 0.9
    point.payload = {
        "research_id": "r1",
        "title": "Title",
        "text": "Body",
        "metadata": {},
    }

    response = MagicMock()
    response.points = [point]
    mock_client.query_points = AsyncMock(return_value=response)

    service = QdrantHybridSearchService(
        mock_client,
        MockDenseEmbedder(),
        MockSparseEncoder(),
    )
    results = await service.search(SearchQuery(query="test query", limit=3))
    assert len(results) == 1
    assert results[0].chunk.research_id == "r1"
    assert results[0].search_type == SearchType.HYBRID
