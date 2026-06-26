from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import RedirectResponse, StreamingResponse

from research_shared.domain.models import DocumentListItem, DocumentRecord, IngestStatus, ResearchChunk
from research_shared.logging_config import get_logger

from core_api.dependencies import get_app_state

router = APIRouter()
logger = get_logger(__name__)


def _validate_pdf(file: UploadFile) -> None:
    filename = file.filename or ""
    is_pdf = filename.lower().endswith(".pdf") or file.content_type == "application/pdf"
    if not is_pdf:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are accepted",
        )


async def _get_record_by_research_id(state, research_id: str) -> DocumentRecord | None:
    if state.ingestion_pipeline is None:
        return None
    records = await state.ingestion_pipeline._state_store.list()
    for record in records:
        if record.research_id == research_id:
            return record
    return None


async def _resolve_pdf_filename(state, research_id: str) -> str | None:
    record = await _get_record_by_research_id(state, research_id)
    if record is not None:
        return record.filename

    for info in state.archive_storage.list_pdfs():
        try:
            described = state.archive_storage.describe(info.filename)
        except FileNotFoundError:
            continue
        if described.research_id == research_id:
            return info.filename
    return None


def _record_to_list_item(record: DocumentRecord) -> DocumentListItem:
    return DocumentListItem(
        research_id=record.research_id,
        filename=record.filename,
        display_name=record.display_name or record.filename,
        status=record.status,
        chunk_count=record.chunk_count,
        indexed_at=record.indexed_at,
        source_url=record.source_url,
    )


def _sort_documents(records: list[DocumentRecord]) -> list[DocumentRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.indexed_at is None,
            -(record.indexed_at.timestamp() if record.indexed_at else 0),
            (record.display_name or record.filename).lower(),
        ),
    )


@router.get("")
async def list_documents(
    status: IngestStatus | None = Query(default=None),
    state = Depends(get_app_state),
) -> dict:
    if state.ingestion_pipeline is None:
        return {"documents": []}

    records = await state.ingestion_pipeline._state_store.list()
    if status is not None:
        records = [record for record in records if record.status == status]
    records = _sort_documents(records)
    items = [_record_to_list_item(record) for record in records]

    logger.info(
        "Document list requested",
        extra={
            "count": len(items),
            "status_filter": status.value if status else None,
            "event": "document.list",
        },
    )
    return {"documents": [item.model_dump(mode="json") for item in items]}


@router.get("/files/{research_id}")
async def download_source_file(
    research_id: str,
    state = Depends(get_app_state),
):
    record = await _get_record_by_research_id(state, research_id)
    if record is not None and record.source_url:
        logger.info(
            "Source file redirect",
            extra={
                "research_id": research_id,
                "event": "document.file_redirect",
            },
        )
        return RedirectResponse(url=record.source_url, status_code=status.HTTP_302_FOUND)

    filename = await _resolve_pdf_filename(state, research_id)
    if filename is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source file not found for research_id={research_id}",
        )

    try:
        content = state.archive_storage.read(filename)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source file not found for research_id={research_id}",
        ) from None
    except Exception as exc:
        logger.error(
            "Document storage read failed",
            extra={"research_id": research_id, "event": "document.storage.error"},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document storage temporarily unavailable",
        ) from exc

    logger.info(
        "Source file download",
        extra={
            "research_id": research_id,
            "attachment_name": filename,
            "event": "document.file_download",
        },
    )
    return StreamingResponse(
        iter([content]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("")
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    display_name: str | None = Form(default=None),
    state = Depends(get_app_state),
) -> dict:
    _validate_pdf(file)
    content = await file.read()
    upload_name = file.filename or "document.pdf"
    try:
        stored = state.staging_storage.save(upload_name, content)
    except Exception as exc:
        logger.error(
            "Staging storage save failed",
            extra={"attachment_name": upload_name, "event": "document.staging.error"},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document storage temporarily unavailable",
        ) from exc
    resolved_display_name = (display_name or "").strip() or stored.filename
    logger.info(
        "Document upload received",
        extra={
            "attachment_name": stored.filename,
            "count": len(content),
            "event": "document.upload",
        },
    )

    if state.settings.ingest_sync:
        result = await state.ingestion_pipeline.process(
            stored.filename,
            display_name=resolved_display_name,
        )
        if result.archive_error:
            state.celery_client.enqueue_archive_document(
                stored.filename,
                display_name=resolved_display_name,
            )
        response.status_code = status.HTTP_201_CREATED
        return {
            "filename": stored.filename,
            "display_name": resolved_display_name,
            "research_id": result.research_id,
            "status": result.status,
            "chunks_indexed": result.chunk_count,
            "skipped": result.skipped,
        }

    task_id = state.celery_client.enqueue_index_document(
        stored.filename,
        display_name=resolved_display_name,
    )
    response.status_code = status.HTTP_202_ACCEPTED
    return {
        "task_id": task_id,
        "research_id": stored.research_id,
        "filename": stored.filename,
        "display_name": resolved_display_name,
        "status": "queued",
    }


@router.post("/batch")
async def upload_documents_batch(
    response: Response,
    files: list[UploadFile] = File(...),
    display_names: list[str] | None = Form(default=None),
    state = Depends(get_app_state),
) -> dict:
    jobs: list[dict] = []
    errors: list[dict] = []
    logger.info(
        "Batch document upload received",
        extra={"count": len(files), "event": "document.upload_batch"},
    )

    for index, file in enumerate(files):
        try:
            _validate_pdf(file)
            content = await file.read()
            stored = state.staging_storage.save(file.filename or "document.pdf", content)
            resolved_display_name = stored.filename
            if display_names and index < len(display_names):
                name = (display_names[index] or "").strip()
                if name:
                    resolved_display_name = name

            if state.settings.ingest_sync:
                result = await state.ingestion_pipeline.process(
                    stored.filename,
                    display_name=resolved_display_name,
                )
                if result.archive_error:
                    state.celery_client.enqueue_archive_document(
                        stored.filename,
                        display_name=resolved_display_name,
                    )
                jobs.append(
                    {
                        "filename": stored.filename,
                        "display_name": resolved_display_name,
                        "research_id": result.research_id,
                        "status": result.status,
                        "chunks_indexed": result.chunk_count,
                    }
                )
            else:
                task_id = state.celery_client.enqueue_index_document(
                    stored.filename,
                    display_name=resolved_display_name,
                )
                jobs.append(
                    {
                        "filename": stored.filename,
                        "display_name": resolved_display_name,
                        "task_id": task_id,
                        "research_id": stored.research_id,
                        "status": "queued",
                    }
                )
        except HTTPException as exc:
            errors.append({"filename": file.filename, "error": exc.detail})
        except Exception as exc:  # noqa: BLE001 — isolate a single file's failure
            errors.append({"filename": file.filename, "error": str(exc)})

    response.status_code = status.HTTP_202_ACCEPTED
    return {"jobs": jobs, "errors": errors}


@router.post("/chunks", status_code=status.HTTP_201_CREATED)
async def upsert_chunks(
    chunks: list[ResearchChunk],
    state = Depends(get_app_state),
) -> dict[str, int]:
    count = await state.vector_store.upsert(chunks)
    return {"upserted": count}


@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    state = Depends(get_app_state),
) -> dict:
    return state.celery_client.get_status(task_id)


@router.delete("/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    chunk_id: str,
    state = Depends(get_app_state),
) -> None:
    await state.vector_store.delete_by_ids([chunk_id])


@router.delete("/research/{research_id}", status_code=status.HTTP_200_OK)
async def delete_research_documents(
    research_id: str,
    state = Depends(get_app_state),
) -> dict[str, str]:
    await state.vector_store.delete_by_research_id(research_id)
    return {"research_id": research_id, "status": "deleted"}
