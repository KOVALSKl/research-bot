from contextlib import asynccontextmanager
from dataclasses import dataclass

import redis.asyncio as aioredis
from fastapi import FastAPI

from research_shared.agents.research_agent import ResearchAgent, create_research_agent
from research_shared.config.settings import Settings, get_settings
from research_shared.literature.pdf_service import ExternalPdfService
from research_shared.literature.service import ExternalLiteratureService, create_literature_service
from research_shared.logging_config import get_logger
from research_shared.ingestion.chunker import RecursiveChunker
from research_shared.ingestion.factory import create_archive_storage, create_staging_storage
from research_shared.ingestion.staging_storage import StagingStorage
from research_shared.ingestion.storage_protocol import DocumentStorage
from research_shared.ingestion.pdf_parser import PyMuPDFParser
from research_shared.ingestion.pipeline import IngestionPipeline
from research_shared.ingestion.state_store import QdrantIngestionStateStore
from research_shared.llm.factory import create_llm_provider
from research_shared.rag.service import RagService
from research_shared.storage.embeddings.factory import create_dense_embedder, create_sparse_encoder
from research_shared.storage.qdrant.client_factory import create_qdrant_client
from research_shared.storage.qdrant.collection import ensure_collection
from research_shared.storage.qdrant.hybrid_search import QdrantHybridSearchService
from research_shared.storage.qdrant.store import QdrantVectorStore

from core_api.api.routes import agent, ask, documents, health, literature, search
from core_api.celery_client import CeleryClient
from core_api.logging_setup import setup_logging
from core_api.middleware.logging import RequestLoggingMiddleware

logger = get_logger(__name__)


@dataclass
class AppState:
    settings: Settings
    vector_store: QdrantVectorStore
    hybrid_search: QdrantHybridSearchService
    celery_client: CeleryClient
    staging_storage: StagingStorage
    archive_storage: DocumentStorage
    file_storage: DocumentStorage
    ingestion_pipeline: IngestionPipeline
    rag_service: RagService
    literature_service: ExternalLiteratureService
    external_pdf_service: ExternalPdfService
    research_agent: ResearchAgent


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = create_qdrant_client(settings)
        await ensure_collection(client, settings)

        dense_embedder = create_dense_embedder(settings)
        sparse_encoder = create_sparse_encoder(settings)

        vector_store = QdrantVectorStore(client, dense_embedder, sparse_encoder, settings)
        state_store = QdrantIngestionStateStore(client, settings)
        await state_store.ensure_collection()
        staging_storage = create_staging_storage(settings)
        archive_storage = create_archive_storage(settings)
        ingestion_pipeline = IngestionPipeline(
            parser=PyMuPDFParser(),
            chunker=RecursiveChunker(settings),
            vector_store=vector_store,
            state_store=state_store,
            staging_storage=staging_storage,
            archive_storage=archive_storage,
            settings=settings,
        )
        hybrid_search = QdrantHybridSearchService(
            client, dense_embedder, sparse_encoder, settings
        )
        rag_service = RagService(hybrid_search, create_llm_provider(settings))
        redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
        literature_service = create_literature_service(settings, redis_client=redis_client)
        external_pdf_service = ExternalPdfService(settings)
        research_agent = create_research_agent(
            settings,
            hybrid_search,
            rag_service,
            literature_service,
            create_llm_provider(settings),
        )

        app.state.container = AppState(
            settings=settings,
            vector_store=vector_store,
            hybrid_search=hybrid_search,
            celery_client=CeleryClient(settings),
            staging_storage=staging_storage,
            archive_storage=archive_storage,
            file_storage=archive_storage,
            ingestion_pipeline=ingestion_pipeline,
            rag_service=rag_service,
            literature_service=literature_service,
            external_pdf_service=external_pdf_service,
            research_agent=research_agent,
        )
        logger.info(
            "Core API started",
            extra={
                "event": "api.start",
                "storage_backend": settings.storage_backend,
                "ingest_staging_dir": settings.ingest_staging_dir,
            },
        )
        yield
        await literature_service.aclose()
        await client.close()
        logger.info("Core API stopped", extra={"event": "api.stop"})

    app = FastAPI(
        title="Research Bot Core API",
        description="Hybrid search API for scientific research RAG",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(health.router)
    app.include_router(search.router, prefix="/search", tags=["search"])
    app.include_router(documents.router, prefix="/documents", tags=["documents"])
    app.include_router(ask.router, prefix="/ask", tags=["ask"])
    app.include_router(agent.router, prefix="/agent", tags=["agent"])
    app.include_router(literature.router, prefix="/literature", tags=["literature"])

    return app
