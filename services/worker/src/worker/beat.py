import asyncio

from research_shared.config.settings import Settings, get_settings
from research_shared.domain.models import IngestStatus
from research_shared.ingestion.file_storage import FileStorage
from research_shared.ingestion.state_store import QdrantIngestionStateStore
from research_shared.storage.qdrant.client_factory import create_qdrant_client

from worker.celery_app import app
from worker.tasks import index_document


@app.task(name="worker.beat.scan_researches")
def scan_researches() -> dict:
    """Periodically enqueue indexing for new/changed files in ``researches/``.

    Thin layer over ``ingestion_state``: lists ``*.pdf``, compares content
    hashes, enqueues ``index_document`` only for new or modified files. Whole
    task is a no-op unless ``researches_scan_enabled`` is true.
    """
    settings = get_settings()
    if not settings.researches_scan_enabled:
        return {"enabled": False, "enqueued": 0}
    return asyncio.run(_scan(settings))


async def _scan(settings: Settings) -> dict:
    file_storage = FileStorage(settings)
    client = create_qdrant_client(settings)
    enqueued: list[str] = []
    paths = file_storage.list()
    try:
        state_store = QdrantIngestionStateStore(client, settings)
        await state_store.ensure_collection()

        for path in paths:
            stored = file_storage.describe(path)
            record = await state_store.get(stored.filename)
            is_indexed = (
                record is not None
                and record.content_hash == stored.content_hash
                and record.status == IngestStatus.INDEXED
            )
            if is_indexed:
                continue
            index_document.delay(str(path))
            enqueued.append(stored.filename)
    finally:
        await client.close()

    return {"enabled": True, "scanned": len(paths), "enqueued": len(enqueued), "files": enqueued}
