from research_shared.config.settings import Settings
from research_shared.domain.models import ResearchChunk, SearchQuery, SearchResult, SearchType


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.qdrant_host == "localhost"
    assert settings.qdrant_port == 6333
    assert settings.qdrant_collection_name == "research_chunks"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.dense_embedding_provider == "ollama"


def test_research_chunk_defaults() -> None:
    chunk = ResearchChunk(
        research_id="paper-1",
        title="Test Paper",
        text="Sample abstract text.",
    )
    assert chunk.id
    assert chunk.metadata == {}


def test_search_query_validation() -> None:
    query = SearchQuery(query="electrolyte transport", limit=5)
    assert query.search_type == SearchType.HYBRID
    assert query.limit == 5


def test_search_result_model() -> None:
    chunk = ResearchChunk(
        id="1",
        research_id="paper-1",
        title="T",
        text="body",
    )
    result = SearchResult(chunk=chunk, score=0.85)
    assert result.search_type == SearchType.HYBRID
