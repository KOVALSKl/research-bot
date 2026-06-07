from fastapi import APIRouter, Depends

from research_shared.domain.models import SearchQuery, SearchResult

from core_api.dependencies import get_app_state

router = APIRouter()


@router.post("", response_model=list[SearchResult])
async def search_documents(
    query: SearchQuery,
    state = Depends(get_app_state),
) -> list[SearchResult]:
    return await state.hybrid_search.search(query)
