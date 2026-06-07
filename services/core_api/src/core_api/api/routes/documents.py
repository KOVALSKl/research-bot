from fastapi import APIRouter, Depends, HTTPException, status

from research_shared.domain.models import ResearchChunk

from core_api.dependencies import get_app_state

router = APIRouter()


@router.post("", status_code=status.HTTP_201_CREATED)
async def upsert_documents(
    chunks: list[ResearchChunk],
    state = Depends(get_app_state),
) -> dict[str, int]:
    count = await state.vector_store.upsert(chunks)
    return {"upserted": count}


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
