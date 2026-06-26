from fastapi import APIRouter, Depends

from core_api.dependencies import get_app_state

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/qdrant")
async def qdrant_health(state=Depends(get_app_state)) -> dict:
    """Diagnostic: Qdrant connection status, vector count, and indexed document count."""
    try:
        collection_info = await state.vector_store._client.get_collection(
            state.settings.qdrant_collection_name
        )
        vectors_count = collection_info.vectors_count or 0
        points_count = collection_info.points_count or 0
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "collection": state.settings.qdrant_collection_name,
        }

    try:
        state_collection_info = await state.vector_store._client.get_collection(
            state.settings.ingest_state_collection
        )
        indexed_docs = state_collection_info.points_count or 0
    except Exception:
        indexed_docs = None

    return {
        "status": "ok",
        "collection": state.settings.qdrant_collection_name,
        "vectors_count": vectors_count,
        "points_count": points_count,
        "indexed_documents": indexed_docs,
        "ingest_state_collection": state.settings.ingest_state_collection,
    }
