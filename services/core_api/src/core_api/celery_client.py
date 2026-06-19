from celery import Celery
from celery.result import AsyncResult

from research_shared.config.settings import Settings

INDEX_DOCUMENT_TASK = "worker.tasks.index_document"

_STATE_MAP = {
    "PENDING": "queued",
    "STARTED": "processing",
    "RETRY": "processing",
    "SUCCESS": "indexed",
    "FAILURE": "failed",
    "REVOKED": "failed",
}


class CeleryClient:
    """Thin Celery producer/status client for core_api.

    Enqueues tasks by name (``send_task``) so the API does not import the
    ``worker`` package, and reads task status via ``AsyncResult``. Broker and
    result backend come from Settings.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or Settings()
        self._app = Celery(
            "core_api",
            broker=settings.effective_celery_broker_url,
            backend=settings.effective_celery_result_backend,
        )

    def enqueue_index_document(
        self,
        path: str,
        *,
        display_name: str | None = None,
    ) -> str:
        kwargs: dict = {}
        if display_name is not None:
            kwargs["display_name"] = display_name
        result = self._app.send_task(INDEX_DOCUMENT_TASK, args=[path], kwargs=kwargs)
        return result.id

    def get_status(self, task_id: str) -> dict:
        result = AsyncResult(task_id, app=self._app)
        state = result.state
        payload: dict = {
            "task_id": task_id,
            "state": state,
            "status": _STATE_MAP.get(state, state.lower()),
        }

        if state == "SUCCESS":
            info = result.result
            if isinstance(info, dict):
                payload["chunks_indexed"] = info.get("chunk_count")
                payload["research_id"] = info.get("research_id")
        elif state == "FAILURE":
            payload["error"] = str(result.result)

        return payload
