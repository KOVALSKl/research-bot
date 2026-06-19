import asyncio

from research_shared.config.settings import get_settings
from research_shared.ingestion.chunker import RecursiveChunker
from research_shared.ingestion.file_storage import FileStorage
from research_shared.ingestion.pdf_parser import PyMuPDFParser
from research_shared.ingestion.pipeline import IngestionPipeline
from research_shared.ingestion.state_store import QdrantIngestionStateStore
from research_shared.storage.embeddings.factory import create_dense_embedder, create_sparse_encoder
from research_shared.storage.qdrant.client_factory import create_qdrant_client
from research_shared.storage.qdrant.collection import ensure_collection
from research_shared.storage.qdrant.store import QdrantVectorStore

from worker.celery_app import app


async def _process(path: str, display_name: str | None = None) -> dict:
    settings = get_settings()
    client = create_qdrant_client(settings)
    try:
        # dense_vector_size from settings avoids probing Ollama on every task.
        await ensure_collection(client, settings, vector_size=settings.dense_vector_size)

        dense_embedder = create_dense_embedder(settings)
        sparse_encoder = create_sparse_encoder(settings)
        pipeline = IngestionPipeline(
            parser=PyMuPDFParser(),
            chunker=RecursiveChunker(settings),
            vector_store=QdrantVectorStore(client, dense_embedder, sparse_encoder, settings),
            state_store=QdrantIngestionStateStore(client, settings),
            file_storage=FileStorage(settings),
            settings=settings,
        )
        result = await pipeline.process(path, display_name=display_name)
        return {
            "filename": result.filename,
            "research_id": result.research_id,
            "status": result.status,
            "chunk_count": result.chunk_count,
            "skipped": result.skipped,
        }
    finally:
        await client.close()


@app.task(name="worker.tasks.index_document")
def index_document(path: str, display_name: str | None = None) -> dict:
    """Process and index a single document. Runs the async pipeline in the
    worker's sync context (heavy embeddings off the API event loop)."""
    return asyncio.run(_process(path, display_name=display_name))


@app.task(name="worker.tasks.index_batch")
def index_batch(
    paths: list[str],
    display_names: list[str] | None = None,
) -> list[dict]:
    names = display_names or [None] * len(paths)
    return [
        asyncio.run(_process(path, display_name=name))
        for path, name in zip(paths, names, strict=False)
    ]
