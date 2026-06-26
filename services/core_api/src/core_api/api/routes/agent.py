from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from research_shared.agents.models import (
    AgentAskRequest,
    AgentAskResponse,
    AgentProgressEvent,
    AgentReasoningEvent,
)
from research_shared.logging_config import get_logger

from core_api.dependencies import get_app_state

router = APIRouter()
logger = get_logger(__name__)


def _resolve_agent_request(
    request: AgentAskRequest,
    default_limit: int,
) -> AgentAskRequest:
    if request.limit is not None:
        return request
    return request.model_copy(update={"limit": default_limit})


@router.post("/ask", response_model=AgentAskResponse)
async def agent_ask(
    request: AgentAskRequest,
    state=Depends(get_app_state),
) -> AgentAskResponse:
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")

    resolved = _resolve_agent_request(request, state.settings.ask_default_limit)
    response = await state.research_agent.run(resolved)

    logger.info(
        "Agent ask completed",
        extra={
            "event": "agent.ask",
            "mode": response.mode,
            "steps_count": len(response.steps),
            "local_sources_count": len(response.sources.local),
            "external_sources_count": len(response.sources.external),
            "source_files_count": len(response.source_files),
        },
    )
    return response


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _agent_stream(resolved: AgentAskRequest, state) -> AsyncIterator[str]:
    queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

    async def on_progress(event: AgentProgressEvent) -> None:
        await queue.put(("progress", event.model_dump(mode="json")))

    async def on_reasoning(event: AgentReasoningEvent) -> None:
        await queue.put(("reasoning", event.model_dump(mode="json")))

    async def run_agent() -> None:
        try:
            response = await state.research_agent.run(
                resolved,
                on_progress=on_progress,
                on_reasoning=on_reasoning,
            )
            await queue.put(("complete", response.model_dump(mode="json")))
        except Exception as exc:
            logger.exception("Agent stream failed", extra={"event": "agent.ask.stream.error"})
            await queue.put(("error", {"detail": str(exc)}))
        finally:
            await queue.put(None)

    task = asyncio.create_task(run_agent())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            event_name, payload = item
            yield _format_sse(event_name, payload)
    finally:
        await task


@router.post("/ask/stream")
async def agent_ask_stream(
    request: AgentAskRequest,
    state=Depends(get_app_state),
) -> StreamingResponse:
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")

    resolved = _resolve_agent_request(request, state.settings.ask_default_limit)
    return StreamingResponse(
        _agent_stream(resolved, state),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
