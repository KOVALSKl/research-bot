from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from research_shared.agents.models import (
    AgentAskRequest,
    AgentAskResponse,
    AgentProgressEvent,
    AgentProgressStage,
    AgentReasoningEvent,
    AgentSources,
    AgentStep,
)
from research_shared.config.settings import Settings

from core_api.api.routes import agent
from core_api.dependencies import get_app_state


def _stream_client() -> tuple[TestClient, list[AgentProgressEvent]]:
    app = FastAPI()
    app.include_router(agent.router, prefix="/agent")
    progress_events: list[AgentProgressEvent] = []

    class FakeAgent:
        async def run(self, request: AgentAskRequest, *, on_progress=None, on_reasoning=None) -> AgentAskResponse:
            if on_progress is not None:
                await on_progress(
                    AgentProgressEvent(
                        stage=AgentProgressStage.CLASSIFY,
                        stage_index=1,
                        stage_total=5,
                        message="classify",
                    )
                )
                await on_progress(
                    AgentProgressEvent(
                        stage=AgentProgressStage.SYNTHESIZE,
                        stage_index=5,
                        stage_total=5,
                        message="synthesize",
                    )
                )
            return AgentAskResponse(
                answer="Stream answer",
                sources=AgentSources(),
                steps=[AgentStep(tool="synthesize", query=request.message, results_count=1)],
            )

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        research_agent=FakeAgent(),
        settings=Settings(_env_file=None),
    )
    return TestClient(app), progress_events


def _parse_sse(body: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    current_event: str | None = None
    data_lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("event:"):
            if current_event and data_lines:
                events.append((current_event, "\n".join(data_lines)))
            current_event = line.split(":", 1)[1].strip()
            data_lines = []
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
        elif line == "" and current_event and data_lines:
            events.append((current_event, "\n".join(data_lines)))
            current_event = None
            data_lines = []
    if current_event and data_lines:
        events.append((current_event, "\n".join(data_lines)))
    return events


def test_agent_stream_emits_reasoning_events() -> None:
    app = FastAPI()
    app.include_router(agent.router, prefix="/agent")

    class FakeAgent:
        async def run(self, request: AgentAskRequest, *, on_progress=None, on_reasoning=None):
            if on_reasoning is not None:
                await on_reasoning(
                    AgentReasoningEvent(
                        iteration=1,
                        max_iterations=6,
                        thought="Searching locally",
                        action="local_hybrid_search",
                        action_summary="Найдено 2 фрагмента",
                    )
                )
            return AgentAskResponse(
                answer="Answer with reasoning",
                sources=AgentSources(),
                steps=[AgentStep(tool="synthesize", query=request.message, results_count=1)],
            )

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        research_agent=FakeAgent(),
        settings=Settings(_env_file=None),
    )
    client = TestClient(app)
    response = client.post("/agent/ask/stream", json={"message": "Question long enough"})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    event_names = [name for name, _ in events]
    assert "reasoning" in event_names
    reasoning_payload = next(data for name, data in events if name == "reasoning")
    assert "Searching locally" in reasoning_payload


def test_agent_stream_returns_progress_and_complete() -> None:
    client, _ = _stream_client()
    response = client.post("/agent/ask/stream", json={"message": "Question long enough"})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    event_names = [name for name, _ in events]
    assert "progress" in event_names
    assert "complete" in event_names
    assert event_names.index("progress") < event_names.index("complete")


def test_agent_stream_complete_has_answer() -> None:
    client, _ = _stream_client()
    response = client.post("/agent/ask/stream", json={"message": "Question long enough"})

    events = _parse_sse(response.text)
    complete_payload = next(data for name, data in events if name == "complete")
    assert "Stream answer" in complete_payload
    assert '"source_files"' in complete_payload


def test_agent_stream_empty_message_422() -> None:
    client, _ = _stream_client()
    response = client.post("/agent/ask/stream", json={"message": "   "})

    assert response.status_code == 422


def test_agent_stream_progress_stage_order() -> None:
    client, _ = _stream_client()
    response = client.post("/agent/ask/stream", json={"message": "Question long enough"})

    events = _parse_sse(response.text)
    progress_payloads = [data for name, data in events if name == "progress"]
    assert progress_payloads[0].find("classify") >= 0
    assert progress_payloads[-1].find("synthesize") >= 0


def test_agent_stream_complete_includes_v15_step_details() -> None:
    app = FastAPI()
    app.include_router(agent.router, prefix="/agent")

    class FakeAgent:
        async def run(self, request: AgentAskRequest, *, on_progress=None, on_reasoning=None) -> AgentAskResponse:
            return AgentAskResponse(
                answer="Detailed answer with [1] and [E1].",
                sources=AgentSources(),
                steps=[
                    AgentStep(
                        tool="query_reformulation",
                        query=request.message,
                        results_count=3,
                        detail=f"{request.message} → ru | en",
                    ),
                    AgentStep(
                        tool="external_literature_search",
                        query="ru | en",
                        results_count=2,
                        detail='{"queries":["ru","en"],"count":2,"fallback_used":false}',
                    ),
                    AgentStep(
                        tool="synthesize",
                        query=request.message,
                        results_count=1,
                        detail="retry=false",
                    ),
                ],
            )

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        research_agent=FakeAgent(),
        settings=Settings(_env_file=None),
    )
    client = TestClient(app)
    response = client.post("/agent/ask/stream", json={"message": "Question long enough"})

    assert response.status_code == 200
    complete_payload = next(data for name, data in _parse_sse(response.text) if name == "complete")
    assert "query_reformulation" in complete_payload
    assert "retry=false" in complete_payload
    assert "fallback_used" in complete_payload


def test_agent_stream_idea_mode_complete_has_assessment() -> None:
    app = FastAPI()
    app.include_router(agent.router, prefix="/agent")

    class FakeAgent:
        async def run(self, request: AgentAskRequest, *, on_progress=None, on_reasoning=None) -> AgentAskResponse:
            from research_shared.agents.models import (
                EvidenceItem,
                IdeaAssessment,
                RelevanceAssessment,
                RelevanceCriterion,
            )

            if on_progress is not None:
                await on_progress(
                    AgentProgressEvent(
                        stage=AgentProgressStage.LOCAL_SEARCH,
                        stage_index=2,
                        stage_total=5,
                        message="Ищу материалы по вашей идее…",
                    )
                )
            return AgentAskResponse(
                mode="idea_evaluation",
                answer="Idea summary with [1].",
                idea_assessment=IdeaAssessment(
                    relevance=RelevanceAssessment(
                        level="medium",
                        criteria=[
                            RelevanceCriterion(
                                name="topic_overlap",
                                level="medium",
                                detail="Relevant [1].",
                            )
                        ],
                        rationale="Relevant [1].",
                    ),
                    evidence_for=[EvidenceItem(text="Pro [1].")],
                    evidence_against=[EvidenceItem(text="Con [E1].")],
                    success_outlook="Outlook [E1].",
                    confidence="medium",
                ),
                sources=AgentSources(),
                steps=[AgentStep(tool="synthesize", query=request.message, results_count=1)],
            )

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        research_agent=FakeAgent(),
        settings=Settings(_env_file=None),
    )
    client = TestClient(app)
    response = client.post(
        "/agent/ask/stream",
        json={
            "message": "Evaluate my idea about membrane materials for water treatment",
            "mode": "idea_evaluation",
        },
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    progress_payloads = [data for name, data in events if name == "progress"]
    assert any("идее" in payload for payload in progress_payloads)
    complete_payload = next(data for name, data in events if name == "complete")
    assert '"idea_assessment"' in complete_payload
    assert '"mode":"idea_evaluation"' in complete_payload.replace(" ", "")
