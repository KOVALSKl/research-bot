import pytest

from research_shared.agents.models import (
    AgentAskRequest,
    AgentAskResponse,
    AgentReasoningEvent,
    AgentSources,
    AgentStep,
    EvidenceItem,
    IdeaAssessment,
    RelevanceAssessment,
    RelevanceCriterion,
)
from research_shared.domain.models import Citation, SourceFileRef
from research_shared.literature.models import ExternalPaper


def _sample_relevance(text: str = "Relevant to [1].") -> RelevanceAssessment:
    return RelevanceAssessment(
        level="medium",
        criteria=[
            RelevanceCriterion(
                name="topic_overlap",
                level="medium",
                detail=text,
            )
        ],
        rationale=text,
    )


def test_agent_ask_request_accepts_idea_evaluation_mode() -> None:
    req = AgentAskRequest(message="Evaluate my idea about ML", mode="idea_evaluation")
    assert req.mode == "idea_evaluation"


def test_agent_step_serializes_thought() -> None:
    step = AgentStep(
        tool="local_hybrid_search",
        query="q",
        results_count=2,
        thought="Search locally first",
    )
    data = step.model_dump()
    assert data["thought"] == "Search locally first"


def test_agent_reasoning_event_serializes() -> None:
    event = AgentReasoningEvent(
        iteration=1,
        max_iterations=6,
        thought="Need external search",
        action="external_literature_search",
        action_summary="Найдено 3 публикаций",
    )
    data = event.model_dump()
    assert data["thought"] == "Need external search"
    assert data["action_summary"] == "Найдено 3 публикаций"


def test_agent_ask_response_serializes_idea_assessment() -> None:
    response = AgentAskResponse(
        mode="idea_evaluation",
        answer="Summary with [1].",
        idea_assessment=IdeaAssessment(
            relevance=_sample_relevance("Relevant to [1]."),
            evidence_for=[EvidenceItem(text="Supports with [1].")],
            evidence_against=[EvidenceItem(text="Gap noted in [E1].")],
            success_outlook="Moderate with [E1].",
            confidence="medium",
        ),
        sources=AgentSources(
            local=[Citation(research_id="r1", title="T", score=0.5)],
            external=[
                ExternalPaper(
                    title="External",
                    url="https://example.org",
                    source="openalex",
                )
            ],
            local_indices=[1],
            external_indices=[1],
        ),
        source_files=[
            SourceFileRef(
                research_id="r1",
                filename="paper.pdf",
                display_name="Paper",
                path="/data/paper.pdf",
            )
        ],
        steps=[AgentStep(tool="synthesize", query="q", results_count=1)],
    )
    data = response.model_dump()
    assert data["mode"] == "idea_evaluation"
    assert data["idea_assessment"]["confidence"] == "medium"
    assert len(data["idea_assessment"]["evidence_for"]) == 1
