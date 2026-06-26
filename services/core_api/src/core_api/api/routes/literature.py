from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from research_shared.literature.models import ExternalPaper
from research_shared.logging_config import get_logger

from core_api.dependencies import get_app_state

router = APIRouter()
logger = get_logger(__name__)


class LiteratureSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int | None = Field(default=None, ge=1, le=50)
    year_from: int | None = None


class LiteratureSearchResponse(BaseModel):
    papers: list[ExternalPaper]
    count: int


def _resolve_limit(request: LiteratureSearchRequest, default_limit: int) -> int:
    return request.limit if request.limit is not None else default_limit


@router.post("/search", response_model=LiteratureSearchResponse)
async def search_literature(
    request: LiteratureSearchRequest,
    state=Depends(get_app_state),
) -> LiteratureSearchResponse:
    limit = _resolve_limit(request, state.settings.literature_default_limit)
    papers = await state.literature_service.search_external(
        query=request.query,
        limit=limit,
        year_from=request.year_from,
    )
    logger.info(
        "Literature search request",
        extra={
            "count": len(papers),
            "event": "literature.api.search",
            "limit": limit,
            "query": request.query,
        },
    )
    return LiteratureSearchResponse(papers=papers, count=len(papers))


@router.get("/papers/{cache_key}/pdf")
async def download_external_pdf(
    cache_key: str,
    pdf_url: str | None = None,
    state=Depends(get_app_state),
) -> StreamingResponse:
    if not cache_key.strip():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid cache key")

    filename = f"{cache_key}.pdf"
    content = state.external_pdf_service.cache.read(cache_key)
    if content is None:
        if not pdf_url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"External PDF not found for cache_key={cache_key}",
            )
        fetched = await state.external_pdf_service.get_or_fetch(cache_key, pdf_url)
        if fetched is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"External PDF fetch failed for cache_key={cache_key}",
            )
        content, filename = fetched

    logger.info(
        "External PDF download",
        extra={
            "cache_key": cache_key,
            "attachment_name": filename,
            "event": "literature.external_pdf_download",
        },
    )
    return StreamingResponse(
        iter([content]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
