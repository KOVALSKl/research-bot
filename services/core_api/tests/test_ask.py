from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from research_shared.config.settings import Settings
from research_shared.domain.models import AskResponse, Citation, ResearchChunk

from core_api.api.routes import ask
from core_api.dependencies import get_app_state


def _client(answer: str | None) -> TestClient:
    app = FastAPI()
    app.include_router(ask.router, prefix="/ask")

    class FakeRag:
        async def ask(self, query):
            return AskResponse(
                answer=answer,
                citations=[Citation(research_id="r1", title="T", page=2, score=0.5)],
                context_chunks=[
                    ResearchChunk(research_id="r1", title="T", text="body", metadata={"page": 2})
                ],
            )

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        rag_service=FakeRag(),
        settings=Settings(_env_file=None),
    )
    return TestClient(app)


def test_ask_without_llm_returns_null_answer() -> None:
    response = _client(answer=None).post("/ask", json={"question": "q", "limit": 3})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] is None
    assert data["citations"][0]["research_id"] == "r1"
    assert data["citations"][0]["page"] == 2
    assert len(data["context_chunks"]) == 1


def test_ask_default_limit_from_settings() -> None:
    app = FastAPI()
    app.include_router(ask.router, prefix="/ask")
    captured: dict = {}
    settings = Settings(_env_file=None, ask_default_limit=15)

    class FakeRag:
        async def ask(self, query):
            captured["limit"] = query.limit
            return AskResponse(answer=None, citations=[], context_chunks=[])

    app.dependency_overrides[get_app_state] = lambda: SimpleNamespace(
        rag_service=FakeRag(),
        settings=settings,
    )
    client = TestClient(app)

    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 200
    assert captured["limit"] == 15


def test_ask_with_llm_returns_answer() -> None:
    response = _client(answer="generated").post("/ask", json={"question": "q", "limit": 3})

    assert response.status_code == 200
    assert response.json()["answer"] == "generated"
