from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from research_shared.agents.models import AgentReasoningEvent, AgentStep
from research_shared.agents.query_reformulation import build_idea_search_queries, build_search_queries
from research_shared.agents.relevance import filter_relevant_results
from research_shared.agents.state import AgentState
from research_shared.agents.tools.external_search import (
    external_literature_search_queries,
    format_external_search_detail,
)
from research_shared.agents.tools.local_search import local_hybrid_search, merge_local_search_results
from research_shared.config.settings import Settings
from research_shared.literature.service import ExternalLiteratureService
from research_shared.llm.prompts import get_agent_react_system_prompt
from research_shared.llm.protocols import LLMProvider
from research_shared.logging_config import get_logger
from research_shared.rag.service import RagService
from research_shared.storage.protocols import HybridSearcher

logger = get_logger(__name__)

WHITELIST_ACTIONS = frozenset(
    {
        "local_hybrid_search",
        "external_literature_search",
        "reformulate_queries",
        "finish",
    }
)

_JSON_BLOCK_PATTERN = re.compile(r"\{[\s\S]*\}")
_TAGGED_THOUGHT = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", re.IGNORECASE | re.DOTALL)
_TAGGED_ACTION = re.compile(r"Action:\s*(\S+)", re.IGNORECASE)
_TAGGED_INPUT = re.compile(r"Action Input:\s*(\{[\s\S]*?\})", re.IGNORECASE | re.DOTALL)

ReasoningCallback = Callable[[AgentReasoningEvent], Awaitable[None] | None]


@dataclass(frozen=True)
class ReactAction:
    thought: str
    action: str
    action_input: dict[str, Any]


@dataclass(frozen=True)
class ReactLoopResult:
    state: AgentState
    forced_finish: bool = False


def parse_react_output(raw: str) -> ReactAction | None:
    text = raw.strip()
    if not text:
        return None

    candidates = [text]
    match = _JSON_BLOCK_PATTERN.search(text)
    if match:
        candidates.insert(0, match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                continue
            action = str(data.get("action", "")).strip()
            thought = str(data.get("thought", "")).strip()
            action_input = data.get("action_input")
            if not isinstance(action_input, dict):
                action_input = {}
            if action:
                return ReactAction(thought=thought, action=action, action_input=action_input)
        except json.JSONDecodeError:
            continue

    thought_match = _TAGGED_THOUGHT.search(text)
    action_match = _TAGGED_ACTION.search(text)
    if action_match is None:
        return None
    action_input: dict[str, Any] = {}
    input_match = _TAGGED_INPUT.search(text)
    if input_match:
        try:
            parsed = json.loads(input_match.group(1))
            if isinstance(parsed, dict):
                action_input = parsed
        except json.JSONDecodeError:
            action_input = {}
    thought = thought_match.group(1).strip() if thought_match else ""
    return ReactAction(
        thought=thought,
        action=action_match.group(1).strip(),
        action_input=action_input,
    )


def _queries_from_input(action_input: dict[str, Any], fallback: list[str]) -> list[str]:
    if "queries" in action_input and isinstance(action_input["queries"], list):
        queries = [str(q).strip() for q in action_input["queries"] if str(q).strip()]
        if queries:
            return queries
    if "query" in action_input:
        query = str(action_input["query"]).strip()
        if query:
            return [query]
    return list(fallback)


async def _emit_reasoning(
    callback: ReasoningCallback | None,
    event: AgentReasoningEvent,
) -> None:
    if callback is None:
        return
    try:
        result = callback(event)
        if result is not None:
            await result
    except Exception:
        logger.exception(
            "Reasoning callback failed",
            extra={"event": "agent.reasoning_callback.error"},
        )


class ReactLoopRunner:
    def __init__(
        self,
        *,
        hybrid_search: HybridSearcher,
        rag_service: RagService,
        literature_service: ExternalLiteratureService,
        llm: LLMProvider | None,
        settings: Settings,
    ) -> None:
        self._searcher = hybrid_search
        self._rag_service = rag_service
        self._literature_service = literature_service
        self._llm = llm
        self._settings = settings

    async def run(
        self,
        state: AgentState,
        *,
        limit: int,
        on_reasoning: ReasoningCallback | None = None,
    ) -> ReactLoopResult:
        forced_finish = False
        system_prompt = get_agent_react_system_prompt(self._settings)

        await self._bootstrap_local_search(state, limit=limit, on_reasoning=on_reasoning)

        while state.iteration < state.max_iterations and not state.finished:
            state.iteration += 1
            raw = self._think(state, system_prompt)
            action = parse_react_output(raw)

            if action is None:
                observation = "Invalid response format. Return JSON with thought, action, action_input."
                state.append_scratchpad("", "parse_error", observation)
                await _emit_reasoning(
                    on_reasoning,
                    AgentReasoningEvent(
                        iteration=state.iteration,
                        max_iterations=state.max_iterations,
                        thought=observation,
                        action=None,
                    ),
                )
                continue

            await _emit_reasoning(
                on_reasoning,
                AgentReasoningEvent(
                    iteration=state.iteration,
                    max_iterations=state.max_iterations,
                    thought=action.thought,
                    action=action.action,
                ),
            )

            if action.action == "finish":
                if not state.has_search_results():
                    observation = (
                        "Cannot finish without search results. "
                        "Run local_hybrid_search or external_literature_search first."
                    )
                    state.append_scratchpad(action.thought, action.action, observation)
                    await _emit_reasoning(
                        on_reasoning,
                        AgentReasoningEvent(
                            iteration=state.iteration,
                            max_iterations=state.max_iterations,
                            thought=action.thought,
                            action=action.action,
                            action_summary=observation,
                        ),
                    )
                    continue
                state.finished = True
                state.append_scratchpad(action.thought, action.action, "Context deemed sufficient.")
                break

            if action.action not in WHITELIST_ACTIONS:
                observation = (
                    f"Unknown action '{action.action}'. "
                    f"Allowed: {', '.join(sorted(WHITELIST_ACTIONS))}."
                )
                state.append_scratchpad(action.thought, action.action, observation)
                await _emit_reasoning(
                    on_reasoning,
                    AgentReasoningEvent(
                        iteration=state.iteration,
                        max_iterations=state.max_iterations,
                        thought=action.thought,
                        action=action.action,
                        action_summary=observation,
                    ),
                )
                continue

            observation, summary = await self._dispatch_action(
                state,
                action,
                limit=limit,
            )
            state.append_scratchpad(action.thought, action.action, observation)
            await _emit_reasoning(
                on_reasoning,
                AgentReasoningEvent(
                    iteration=state.iteration,
                    max_iterations=state.max_iterations,
                    thought=action.thought,
                    action=action.action,
                    action_summary=summary,
                ),
            )

        if not state.finished and state.iteration >= state.max_iterations:
            forced_finish = True
            state.finished = True

        if not state.combined_context.strip():
            await self._safety_net_search(state, limit=limit, on_reasoning=on_reasoning)

        return ReactLoopResult(state=state, forced_finish=forced_finish)

    async def ensure_idea_external_search(
        self,
        state: AgentState,
        *,
        limit: int,
        on_reasoning: ReasoningCallback | None = None,
    ) -> None:
        if state.mode != "idea_evaluation":
            return

        if state.external_papers:
            return

        external_action = ReactAction(
            thought="Для оценки идеи нужны внешние публикации — ищу в научных базах.",
            action="external_literature_search",
            action_input={},
        )
        _, external_summary = await self._tool_external_search(
            state,
            external_action,
            limit=limit,
        )
        state.steps[-1].detail = f"mandatory; {state.steps[-1].detail or ''}".strip("; ")
        await _emit_reasoning(
            on_reasoning,
            AgentReasoningEvent(
                iteration=state.iteration,
                max_iterations=state.max_iterations,
                thought=external_action.thought,
                action=external_action.action,
                action_summary=external_summary,
            ),
        )

    async def _bootstrap_local_search(
        self,
        state: AgentState,
        *,
        limit: int,
        on_reasoning: ReasoningCallback | None,
    ) -> None:
        action = ReactAction(
            thought="Ищу релевантные фрагменты в загруженных PDF.",
            action="local_hybrid_search",
            action_input={},
        )
        observation, summary = await self._tool_local_search(state, action, limit=limit)
        state.steps[-1].detail = f"bootstrap; {state.steps[-1].detail or ''}".strip("; ")
        await _emit_reasoning(
            on_reasoning,
            AgentReasoningEvent(
                iteration=0,
                max_iterations=state.max_iterations,
                thought=action.thought,
                action=action.action,
                action_summary=summary or observation,
            ),
        )

    async def _safety_net_search(
        self,
        state: AgentState,
        *,
        limit: int,
        on_reasoning: ReasoningCallback | None,
    ) -> None:
        local_action = ReactAction(
            thought="Повторный локальный поиск — контекст всё ещё пуст.",
            action="local_hybrid_search",
            action_input={},
        )
        _, local_summary = await self._tool_local_search(state, local_action, limit=limit)
        state.steps[-1].detail = f"safety_net; {state.steps[-1].detail or ''}".strip("; ")
        await _emit_reasoning(
            on_reasoning,
            AgentReasoningEvent(
                iteration=state.iteration,
                max_iterations=state.max_iterations,
                thought=local_action.thought,
                action=local_action.action,
                action_summary=local_summary,
            ),
        )

        if state.combined_context.strip():
            return

        external_action = ReactAction(
            thought="Локальных данных нет — ищу во внешних научных базах.",
            action="external_literature_search",
            action_input={},
        )
        _, external_summary = await self._tool_external_search(
            state,
            external_action,
            limit=limit,
        )
        state.steps[-1].detail = f"safety_net; {state.steps[-1].detail or ''}".strip("; ")
        await _emit_reasoning(
            on_reasoning,
            AgentReasoningEvent(
                iteration=state.iteration,
                max_iterations=state.max_iterations,
                thought=external_action.thought,
                action=external_action.action,
                action_summary=external_summary,
            ),
        )

    def _think(self, state: AgentState, system_prompt: str) -> str:
        provider = self._llm
        if provider is None:
            return ""

        user_content = (
            f"<user_query>\n{state.message}\n</user_query>\n\n"
            f"Current search queries: {state.search_queries or [state.message]}\n\n"
            f"Combined context length: {len(state.combined_context)} chars\n\n"
            f"Scratchpad:\n{state.scratchpad or '(empty)'}\n\n"
            "Choose the next action."
        )
        if hasattr(provider, "_system_prompt"):
            original = provider._system_prompt
            provider._system_prompt = system_prompt
            try:
                return provider.generate(user_content, state.combined_context)
            finally:
                provider._system_prompt = original
        return provider.generate(user_content, state.combined_context)

    async def _dispatch_action(
        self,
        state: AgentState,
        action: ReactAction,
        *,
        limit: int,
    ) -> tuple[str, str]:
        if action.action == "reformulate_queries":
            return await self._tool_reformulate_queries(state, action)
        if action.action == "local_hybrid_search":
            return await self._tool_local_search(state, action, limit=limit)
        if action.action == "external_literature_search":
            return await self._tool_external_search(state, action, limit=limit)
        return "Unhandled action.", ""

    async def _tool_reformulate_queries(
        self,
        state: AgentState,
        action: ReactAction,
    ) -> tuple[str, str]:
        if state.mode == "idea_evaluation":
            queries = await build_idea_search_queries(
                state.message,
                self._llm,
                self._settings,
            )
            detail_prefix = "idea → "
        else:
            queries = await build_search_queries(state.message, self._llm, self._settings)
            detail_prefix = ""

        custom = _queries_from_input(action.action_input, [])
        if custom:
            queries = custom

        state.search_queries = queries
        detail = f"{detail_prefix}{' | '.join(queries)}"
        state.steps.append(
            AgentStep(
                tool="reformulate_queries",
                query=state.message,
                results_count=len(queries),
                detail=detail,
                thought=action.thought or None,
            )
        )
        observation = f"Reformulated {len(queries)} search queries."
        summary = f"Запросы: {detail}"
        return observation, summary

    async def _tool_local_search(
        self,
        state: AgentState,
        action: ReactAction,
        *,
        limit: int,
    ) -> tuple[str, str]:
        queries = _queries_from_input(
            action.action_input,
            state.search_queries or [state.message],
        )
        local_parts = await asyncio.gather(
            *[
                local_hybrid_search(self._searcher, self._rag_service, query, limit)
                for query in queries
            ]
        )
        local = merge_local_search_results(list(local_parts))
        filtered = await filter_relevant_results(
            state.message,
            local,
            self._settings,
            llm=self._llm,
        )

        if filtered.citations:
            state.local_citations = filtered.citations
            state.local_context = filtered.context
            state.rebuild_combined_context()

        query_detail = " | ".join(queries)
        state.steps.append(
            AgentStep(
                tool="local_hybrid_search",
                query=query_detail,
                results_count=len(filtered.results),
                detail=f"{len(local.results)}→{len(filtered.results)} chunks",
                thought=action.thought or None,
            )
        )
        observation = (
            f"Local search returned {len(filtered.results)} relevant chunks "
            f"from {len(queries)} queries."
        )
        summary = f"Найдено {len(filtered.results)} фрагментов в PDF"
        return observation, summary

    async def _tool_external_search(
        self,
        state: AgentState,
        action: ReactAction,
        *,
        limit: int,
    ) -> tuple[str, str]:
        queries = _queries_from_input(
            action.action_input,
            state.search_queries or [state.message],
        )
        external = await external_literature_search_queries(
            self._literature_service,
            queries,
            self._settings.literature_idea_mode_limit
            if state.mode == "idea_evaluation"
            else self._settings.literature_default_limit,
            fallback_query=state.message,
            mode=state.mode,
        )
        if external.papers:
            state.external_papers = external.papers
            state.external_context = external.context
            state.rebuild_combined_context()

        query_detail = " | ".join(queries)
        detail = format_external_search_detail(external)
        state.steps.append(
            AgentStep(
                tool="external_literature_search",
                query=query_detail,
                results_count=len(external.papers),
                detail=detail,
                thought=action.thought or None,
            )
        )
        observation = f"External search returned {len(external.papers)} papers."
        summary = f"Найдено {len(external.papers)} публикаций"
        return observation, summary


async def run_rule_based_fallback(
    state: AgentState,
    *,
    hybrid_search: HybridSearcher,
    rag_service: RagService,
    limit: int,
) -> AgentState:
    local = await local_hybrid_search(
        hybrid_search,
        rag_service,
        state.message,
        limit,
    )
    if local.citations:
        state.local_citations = local.citations
        state.local_context = local.context
        state.rebuild_combined_context()
    state.steps.append(
        AgentStep(
            tool="local_hybrid_search",
            query=state.message,
            results_count=len(local.results),
            detail="llm=disabled",
        )
    )
    state.finished = True
    return state
