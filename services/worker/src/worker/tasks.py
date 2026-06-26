import asyncio

from research_shared.config.settings import get_settings
from research_shared.ingestion.chunker import RecursiveChunker
from research_shared.ingestion.factory import create_archive_storage, create_staging_storage
from research_shared.ingestion.pdf_parser import PyMuPDFParser
from research_shared.ingestion.pipeline import IngestionPipeline
from research_shared.ingestion.state_store import QdrantIngestionStateStore
from research_shared.storage.embeddings.factory import create_dense_embedder, create_sparse_encoder
from research_shared.storage.qdrant.client_factory import create_qdrant_client
from research_shared.storage.qdrant.collection import ensure_collection
from research_shared.storage.qdrant.store import QdrantVectorStore

from worker.celery_app import app


def _build_pipeline() -> tuple[IngestionPipeline, object]:
    settings = get_settings()
    client = create_qdrant_client(settings)
    return (
        IngestionPipeline(
            parser=PyMuPDFParser(),
            chunker=RecursiveChunker(settings),
            vector_store=QdrantVectorStore(
                client,
                create_dense_embedder(settings),
                create_sparse_encoder(settings),
                settings,
            ),
            state_store=QdrantIngestionStateStore(client, settings),
            staging_storage=create_staging_storage(settings),
            archive_storage=create_archive_storage(settings),
            settings=settings,
        ),
        client,
    )


async def _process(filename: str, display_name: str | None = None) -> dict:
    settings = get_settings()
    pipeline, client = _build_pipeline()
    try:
        await ensure_collection(client, settings, vector_size=settings.dense_vector_size)
        result = await pipeline.process(filename, display_name=display_name)
        payload = {
            "filename": result.filename,
            "research_id": result.research_id,
            "status": result.status,
            "chunk_count": result.chunk_count,
            "skipped": result.skipped,
        }
        if result.archive_error:
            archive_document.delay(filename, display_name=display_name)
            payload["archive_error"] = result.archive_error
            payload["archive_retry_enqueued"] = True
        return payload
    finally:
        await client.close()


async def _archive(filename: str, display_name: str | None = None) -> dict:
    settings = get_settings()
    pipeline, client = _build_pipeline()
    try:
        await ensure_collection(client, settings, vector_size=settings.dense_vector_size)
        result = await pipeline.archive_only(filename, display_name=display_name)
        payload = {
            "filename": result.filename,
            "research_id": result.research_id,
            "status": result.status,
            "chunk_count": result.chunk_count,
        }
        if result.archive_error:
            payload["archive_error"] = result.archive_error
        return payload
    finally:
        await client.close()


@app.task(name="worker.tasks.index_document")
def index_document(filename: str, display_name: str | None = None) -> dict:
    """Process and index a single document. Runs the async pipeline in the
    worker's sync context (heavy embeddings off the API event loop)."""
    return asyncio.run(_process(filename, display_name=display_name))


@app.task(name="worker.tasks.archive_document", bind=True, max_retries=3, default_retry_delay=60)
def archive_document(self, filename: str, display_name: str | None = None) -> dict:
    """Retry archive for an indexed document whose staging file is still available."""
    try:
        return asyncio.run(_archive(filename, display_name=display_name))
    except Exception as exc:  # noqa: BLE001 — Celery retry for transient archive errors
        raise self.retry(exc=exc) from exc


@app.task(name="worker.tasks.index_batch")
def index_batch(
    filenames: list[str],
    display_names: list[str] | None = None,
) -> list[dict]:
    names = display_names or [None] * len(filenames)
    return [
        asyncio.run(_process(filename, display_name=name))
        for filename, name in zip(filenames, names, strict=False)
    ]
