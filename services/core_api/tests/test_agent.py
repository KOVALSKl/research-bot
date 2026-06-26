from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from research_shared.agents.models import AgentAskRequest, AgentAskResponse, AgentSources, AgentStep
from research_shared.config.settings import Settings

from core_api.api.routes import agent
from core_api.dependencies import get_app_state


def _client(
    *,
    settings: Settings | None = None,
) -> tuple[TestClient, dict]:
    app = FastAPI()
    app.include_router(agent.router, prefix="/agent")
    captured: dict = {}
    settings = settings or Settings(_env_file=None)

    class FakeAgent:
        async def run(self, request: AgentAskRequest, *, on_progress=None) -> AgentAskResponse:
            captured["request"] = request
            return AgentAskResponse(
                answer="Agent answer",
                sources=AgentSources(),
                steps=[
                    AgentStep(tool="classify", query=request.message, results_count=1),
                    AgentStep(tool="synthesize", query=request.message, results_count=1),
                ],
            )

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        research_agent=FakeAgent(),
        settings=settings,
    )
    return TestClient(app), captured


def test_agent_ask_returns_response() -> None:
    client, captured = _client()
    response = client.post(
        "/agent/ask",
        json={"message": "What is AI?", "mode": "question", "limit": 5},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Agent answer"
    assert data["mode"] == "question"
    assert len(data["steps"]) == 2
    assert captured["request"].limit == 5


def test_agent_ask_default_limit_from_settings() -> None:
    settings = Settings(_env_file=None, ask_default_limit=12)
    client, captured = _client(settings=settings)

    response = client.post("/agent/ask", json={"message": "Question long enough"})

    assert response.status_code == 200
    assert captured["request"].limit == 12


def test_agent_ask_empty_message_returns_422() -> None:
    client, _ = _client()
    response = client.post("/agent/ask", json={"message": "   "})

    assert response.status_code == 422


def test_agent_ask_invalid_mode_rejected_by_schema() -> None:
    with pytest.raises(ValidationError):
        AgentAskRequest(message="hello", mode="invalid_mode")


def test_agent_ask_idea_evaluation_mode_accepted() -> None:
    client, captured = _client()
    response = client.post(
        "/agent/ask",
        json={"message": "Evaluate my research idea about ML", "mode": "idea_evaluation"},
    )

    assert response.status_code == 200
    assert captured["request"].mode == "idea_evaluation"


def test_agent_ask_auto_mode_accepted() -> None:
    client, captured = _client()
    response = client.post("/agent/ask", json={"message": "Question long enough", "mode": "auto"})

    assert response.status_code == 200
    assert captured["request"].mode == "auto"


def test_agent_ask_response_contains_sources_and_steps() -> None:
    client, _ = _client()
    response = client.post("/agent/ask", json={"message": "Question long enough"})

    data = response.json()
    assert "sources" in data
    assert "local" in data["sources"]
    assert "external" in data["sources"]
    assert "local_indices" in data["sources"]
    assert "external_indices" in data["sources"]
    assert "source_files" in data
    assert data["idea_assessment"] is None
