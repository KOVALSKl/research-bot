"""Legacy RAG endpoint. Prefer ``POST /agent/ask`` for the Research Agent pipeline."""

from fastapi import APIRouter, Depends

from research_shared.domain.models import AskQuery, AskResponse
from research_shared.logging_config import get_logger

from core_api.dependencies import get_app_state

router = APIRouter()
logger = get_logger(__name__)


def _resolve_ask_query(query: AskQuery, default_limit: int) -> AskQuery:
    if query.limit is not None:
        return query
    return query.model_copy(update={"limit": default_limit})


@router.post("", response_model=AskResponse)
async def ask_question(
    query: AskQuery,
    state = Depends(get_app_state),
) -> AskResponse:
    resolved = _resolve_ask_query(query, state.settings.ask_default_limit)
    response = await state.rag_service.ask(resolved)
    logger.info(
        "Ask query received",
        extra={
            "citations_count": len(response.citations),
            "event": "ask.query",
            "source_files_count": len(response.source_files),
        },
    )
    return response
