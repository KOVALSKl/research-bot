from __future__ import annotations

from dataclasses import dataclass, field

from research_shared.agents.context_builder import build_agent_context
from research_shared.agents.models import AgentStep, ResolvedAgentMode
from research_shared.domain.models import Citation
from research_shared.literature.models import ExternalPaper


@dataclass
class AgentState:
    message: str
    mode: ResolvedAgentMode
    max_iterations: int
    search_queries: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    local_citations: list[Citation] = field(default_factory=list)
    external_papers: list[ExternalPaper] = field(default_factory=list)
    local_context: str = ""
    external_context: str = ""
    combined_context: str = ""
    scratchpad: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    iteration: int = 0
    finished: bool = False

    def rebuild_combined_context(self) -> None:
        self.combined_context = build_agent_context(self.local_context, self.external_context)

    def has_search_results(self) -> bool:
        return bool(self.local_citations) or bool(self.external_papers)

    def local_chunk_count(self) -> int:
        return len(self.local_citations)

    def append_scratchpad(self, thought: str, action: str, observation: str) -> None:
        block = (
            f"Thought: {thought}\n"
            f"Action: {action}\n"
            f"Observation: {observation}\n"
        )
        self.scratchpad = f"{self.scratchpad}\n{block}".strip() if self.scratchpad else block
