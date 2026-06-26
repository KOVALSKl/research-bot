from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from research_shared.domain.models import Citation, ExternalSourceFileRef, SourceFileRef
from research_shared.literature.models import ExternalPaper


class AgentProgressStage(str, Enum):
    CLASSIFY = "classify"
    LOCAL_SEARCH = "local_search"
    RELEVANCE_FILTER = "relevance_filter"
    EXTERNAL_SEARCH = "external_search"
    SYNTHESIZE = "synthesize"


class ExternalPaperPreview(BaseModel):
    title: str
    url: str


class AgentProgressEvent(BaseModel):
    stage: AgentProgressStage
    stage_index: int
    stage_total: int = 5
    message: str
    external_preview: list[ExternalPaperPreview] | None = None


class AgentReasoningEvent(BaseModel):
    iteration: int
    max_iterations: int
    thought: str
    action: str | None = None
    action_summary: str | None = None


class AgentStep(BaseModel):
    tool: str
    query: str
    results_count: int
    thought: str | None = None
    detail: str | None = None
    filtered_count: int | None = None


AgentMode = Literal["question", "idea_evaluation", "auto"]
ResolvedAgentMode = Literal["question", "idea_evaluation"]


class AgentAskRequest(BaseModel):
    message: str
    mode: AgentMode = "auto"
    limit: int | None = Field(default=None, ge=1, le=50)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    text: str


RelevanceLevel = Literal["low", "medium", "high"]
RelevanceCriterionName = Literal[
    "local_sources",
    "external_publications",
    "topic_overlap",
]


class RelevanceCriterion(BaseModel):
    name: RelevanceCriterionName
    level: RelevanceLevel
    detail: str


class RelevanceAssessment(BaseModel):
    level: RelevanceLevel
    criteria: list[RelevanceCriterion] = Field(min_length=1, max_length=3)
    rationale: str


class IdeaAssessment(BaseModel):
    relevance: RelevanceAssessment
    evidence_for: list[EvidenceItem] = Field(default_factory=list)
    evidence_against: list[EvidenceItem] = Field(default_factory=list)
    success_outlook: str
    confidence: Literal["low", "medium", "high"]


class AgentSources(BaseModel):
    local: list[Citation] = Field(default_factory=list)
    external: list[ExternalPaper] = Field(default_factory=list)
    local_indices: list[int] = Field(default_factory=list)
    external_indices: list[int] = Field(default_factory=list)


class AgentAskResponse(BaseModel):
    mode: ResolvedAgentMode = "question"
    answer: str
    idea_assessment: IdeaAssessment | None = None
    sources: AgentSources = Field(default_factory=AgentSources)
    source_files: list[SourceFileRef] = Field(default_factory=list)
    external_source_files: list[ExternalSourceFileRef] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
