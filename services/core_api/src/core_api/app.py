from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from research_shared.config.settings import Settings, get_settings
from research_shared.storage.embeddings.factory import create_dense_embedder, create_sparse_encoder
from research_shared.storage.qdrant.client_factory import create_qdrant_client
from research_shared.storage.qdrant.collection import ensure_collection
from research_shared.storage.qdrant.hybrid_search import QdrantHybridSearchService
from research_shared.storage.qdrant.store import QdrantVectorStore

from core_api.api.routes import documents, health, search


@dataclass
class AppState:
    settings: Settings
    vector_store: QdrantVectorStore
    hybrid_search: QdrantHybridSearchService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = create_qdrant_client(settings)
        await ensure_collection(client, settings)

        dense_embedder = create_dense_embedder(settings)
        sparse_encoder = create_sparse_encoder(settings)

        app.state.container = AppState(
            settings=settings,
            vector_store=QdrantVectorStore(client, dense_embedder, sparse_encoder, settings),
            hybrid_search=QdrantHybridSearchService(
                client, dense_embedder, sparse_encoder, settings
            ),
        )
        yield
        await client.close()

    app = FastAPI(
        title="Research Bot Core API",
        description="Hybrid search API for scientific research RAG",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(search.router, prefix="/search", tags=["search"])
    app.include_router(documents.router, prefix="/documents", tags=["documents"])

    return app
