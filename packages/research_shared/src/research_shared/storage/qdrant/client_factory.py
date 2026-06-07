from qdrant_client import AsyncQdrantClient

from research_shared.config.settings import Settings


def create_qdrant_client(settings: Settings | None = None) -> AsyncQdrantClient:
    settings = settings or Settings()
    return AsyncQdrantClient(url=settings.qdrant_url)
