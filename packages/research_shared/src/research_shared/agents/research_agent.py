from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from research_shared.agents.classify import classify_request
from research_shared.agents.diagnostics import (
    build_empty_context_message,
    build_search_attestation,
    search_sufficient,
)
from research_shared.agents.models import (
    AgentAskRequest,
    AgentAskResponse,
    AgentProgressEvent,
    AgentProgressStage,
    AgentReasoningEvent,
    AgentSources,
    AgentStep,
    EvidenceItem,
    ExternalPaperPreview,
    IdeaAssessment,
    RelevanceAssessment,
    RelevanceCriterion,
)
from research_shared.agents.query_reformulation import build_idea_search_queries, build_search_queries
from research_shared.agents.react_loop import ReactLoopRunner, run_rule_based_fallback
from research_shared.agents.state import AgentState
from research_shared.config.settings import Settings
from research_shared.literature.service import ExternalLiteratureService
from research_shared.llm.prompts import (
    AGENT_QUESTION_RETRY_SUFFIX,
    get_agent_idea_eval_prompt,
    get_agent_question_prompt,
)
from research_shared.llm.protocols import LLMProvider
from research_shared.logging_config import get_logger
from research_shared.rag.cited_sources import extract_cited_sources
from research_shared.rag.service import RagService
from research_shared.storage.protocols import HybridSearcher

logger = get_logger(__name__)

_LLM_DISABLED_MESSAGE = "LLM отключён. Найдены релевантные фрагменты в локальных источниках."

ProgressCallback = Callable[[AgentProgressEvent], Awaitable[None] | None]
ReasoningCallback = Callable[[AgentReasoningEvent], Awaitable[None] | None]

_CLASSIFY_PROGRESS_MESSAGE = "Определяю тип запроса…"
_SYNTHESIZE_PROGRESS_MESSAGE = "Формирую ответ…"
_IDEA_SYNTHESIZE_PROGRESS_MESSAGE = "Оцениваю идею на основе литературы…"
_PROGRESS_STAGE_TOTAL = 2

_SHALLOW_META_PATTERNS = (
    "см. источник",
    "смотрите источник",
    "информация представлена",
    "ответ есть в",
    "можно найти в",
    "подробнее в",
    "аргумент за",
    "конкретным фактом",
    "ответ содержится",
    "информация в [",
    "информация содержится",
    "содержится в [",
    "см. [",
    "смотрите [",
    "подробности в",
    "детали в [",
)

_MIN_EVIDENCE_SUBSTANCE_LENGTH = 40

_CITATION_PATTERN = re.compile(r"\[(?:E)?\d+\]")
_EXTERNAL_CITATION_PATTERN = re.compile(r"\[E\d+\]")
_JSON_BLOCK_PATTERN = re.compile(r"\{[\s\S]*\}")
_IDEA_EVIDENCE_RETRY_SUFFIX = """\
В предыдущем JSON элементы evidence_for и evidence_against были шаблонными или без содержания. \
Перепиши JSON: каждый элемент — полное предложение с конкретным фактом из контекста \
и inline-цитатой [n] или [En]. Не используй фразы «аргумент за» или «информация в [»."""


class _IdeaEvalPayload(BaseModel):
    # Simplified flat schema (new prompt)
    relevance_level: str | None = None
    relevance_rationale: str | None = None
    # Legacy nested schema (old prompt — kept for backward compatibility)
    relevance: Any = None
    evidence_for: list[str]
    evidence_against: list[str]
    success_outlook: str
    confidence: str
    summary: str = ""


class ResearchAgent:
    """Stateful ReAct research agent: classify → tool loop → synthesize."""

    def __init__(
        self,
        hybrid_search: HybridSearcher,
        rag_service: RagService,
        literature_service: ExternalLiteratureService,
        llm_provider: LLMProvider | None,
        settings: Settings,
    ) -> None:
        self._searcher = hybrid_search
        self._rag_service = rag_service
        self._literature_service = literature_service
        self._llm = llm_provider
        self._settings = settings

    async def run(
        self,
        request: AgentAskRequest,
        *,
        on_progress: ProgressCallback | None = None,
        on_reasoning: ReasoningCallback | None = None,
    ) -> AgentAskResponse:
        return await self._run_react_pipeline(
            request,
            on_progress=on_progress,
            on_reasoning=on_reasoning,
        )

    async def _run_react_pipeline(
        self,
        request: AgentAskRequest,
        *,
        on_progress: ProgressCallback | None = None,
        on_reasoning: ReasoningCallback | None = None,
    ) -> AgentAskResponse:
        limit = request.limit or self._settings.ask_default_limit
        message = request.message.strip()
        steps: list[AgentStep] = []

        logger.info(
            "Research agent ReAct pipeline started",
            extra={
                "event": "agent.run.start",
                "mode": request.mode,
                "react": True,
                "limit": limit,
            },
        )

        await self._emit_progress(
            on_progress,
            AgentProgressStage.CLASSIFY,
            _CLASSIFY_PROGRESS_MESSAGE,
            stage_index=1,
        )

        resolved_mode = classify_request(
            request,
            settings=self._settings,
            llm=self._llm,
        )
        steps.append(
            AgentStep(
                tool="classify",
                query=message,
                results_count=1,
                detail=resolved_mode,
            )
        )

        if resolved_mode == "idea_evaluation":
            initial_queries = await build_idea_search_queries(message, self._llm, self._settings)
            if len(initial_queries) > 1:
                steps.append(
                    AgentStep(
                        tool="query_reformulation",
                        query=message,
                        results_count=len(initial_queries),
                        detail=" | ".join(initial_queries),
                    )
                )
        else:
            initial_queries = await build_search_queries(message, self._llm, self._settings)

        if request.conversation_history:
            history_lines = []
            for turn in request.conversation_history:
                role = "Пользователь" if turn.get("role") == "user" else "Ассистент"
                history_lines.append(f"{role}: {turn.get('content', '')}")
            history_text = "\n".join(history_lines)
            effective_message = f"[История диалога]\n{history_text}\n\n[Текущий вопрос]\n{message}"
        else:
            effective_message = message

        state = AgentState(
            message=effective_message,
            mode=resolved_mode,
            max_iterations=self._settings.agent_max_iterations,
            search_queries=initial_queries or [message],
        )

        if self._llm is None:
            state = await run_rule_based_fallback(
                state,
                hybrid_search=self._searcher,
                rag_service=self._rag_service,
                limit=limit,
            )
            steps.extend(state.steps)
        else:
            runner = ReactLoopRunner(
                hybrid_search=self._searcher,
                rag_service=self._rag_service,
                literature_service=self._literature_service,
                llm=self._llm,
                settings=self._settings,
            )
            result = await runner.run(state, limit=limit, on_reasoning=on_reasoning)
            state = result.state
            await runner.ensure_idea_external_search(
                state,
                limit=limit,
                on_reasoning=on_reasoning,
            )
            steps.extend(state.steps)

        synthesize_message = (
            _IDEA_SYNTHESIZE_PROGRESS_MESSAGE
            if resolved_mode == "idea_evaluation"
            else _SYNTHESIZE_PROGRESS_MESSAGE
        )
        await self._emit_progress(
            on_progress,
            AgentProgressStage.SYNTHESIZE,
            synthesize_message,
            stage_index=2,
        )

        combined_context = state.combined_context
        local_context = state.local_context

        attestation = build_search_attestation(
            state.steps,
            local_chunks=state.local_chunk_count(),
            external_papers=len(state.external_papers),
        )

        if not search_sufficient(attestation, resolved_mode, self._settings) or not combined_context.strip():
            answer = build_empty_context_message(
                mode=resolved_mode,
                llm_enabled=self._settings.llm_enabled or self._llm is not None,
                llm_available=self._llm is not None,
                attestation=attestation,
            )
            idea_assessment = None
            synthesis_detail = f"context=empty;{attestation.format_detail()}"
        elif resolved_mode == "idea_evaluation":
            answer, idea_assessment, synthesis_detail = self._synthesize_idea(
                message,
                combined_context,
                local_context,
                external_papers=state.external_papers,
            )
        else:
            idea_assessment = None
            answer, synthesis_detail = self._synthesize(message, combined_context, local_context)
            if synthesis_detail:
                synthesis_detail = f"{synthesis_detail};{attestation.format_detail()}"
            else:
                synthesis_detail = attestation.format_detail()

        steps.append(
            AgentStep(
                tool="synthesize",
                query=message,
                results_count=1 if answer else 0,
                detail=synthesis_detail,
            )
        )

        cited = extract_cited_sources(answer, state.local_citations, state.external_papers)
        response = AgentAskResponse(
            mode=resolved_mode,
            answer=answer,
            idea_assessment=idea_assessment,
            sources=AgentSources(
                local=cited.local,
                external=cited.external,
                local_indices=cited.local_indices,
                external_indices=cited.external_indices,
            ),
            source_files=cited.source_files,
            external_source_files=cited.external_source_files,
            steps=steps,
        )

        logger.info(
            "Research agent ReAct pipeline finished",
            extra={
                "event": "agent.run.finish",
                "resolved_mode": resolved_mode,
                "react": True,
                "steps_count": len(steps),
                "iterations": state.iteration,
                "cited_local_count": len(cited.local),
                "cited_external_count": len(cited.external),
                "source_files_count": len(cited.source_files),
                "external_source_files_count": len(cited.external_source_files),
            },
        )
        return response

    async def _emit_progress(
        self,
        on_progress: ProgressCallback | None,
        stage: AgentProgressStage,
        message: str,
        *,
        stage_index: int,
        external_preview: list[ExternalPaperPreview] | None = None,
    ) -> None:
        if on_progress is None:
            return
        event = AgentProgressEvent(
            stage=stage,
            stage_index=stage_index,
            stage_total=_PROGRESS_STAGE_TOTAL,
            message=message,
            external_preview=external_preview,
        )
        try:
            result = on_progress(event)
            if result is not None:
                await result
        except Exception:
            logger.exception(
                "Progress callback failed",
                extra={"event": "agent.progress_callback.error", "stage": stage.value},
            )

    def _synthesize(
        self,
        message: str,
        combined_context: str,
        local_context: str,
    ) -> tuple[str, str | None]:
        if self._llm is None:
            return self._fallback_answer(local_context, combined_context), None

        system_prompt = get_agent_question_prompt(self._settings)
        raw_answer = self._generate_with_prompt(
            message,
            combined_context,
            system_prompt,
        )
        answer = RagService._dedupe_answer(raw_answer)

        if (
            self._settings.agent_synthesis_retry_on_shallow
            and combined_context.strip()
            and self._is_shallow_answer(answer)
        ):
            retry_prompt = f"{system_prompt}\n\n{AGENT_QUESTION_RETRY_SUFFIX}"
            retry_answer = self._generate_with_prompt(
                message,
                combined_context,
                retry_prompt,
            )
            answer = RagService._dedupe_answer(retry_answer)
            return answer, "retry=true"

        return answer, "retry=false"

    def _synthesize_idea(
        self,
        message: str,
        combined_context: str,
        local_context: str,
        *,
        external_papers: list | None = None,
    ) -> tuple[str, IdeaAssessment | None, str | None]:
        if self._llm is None:
            fallback = self._fallback_answer(local_context, combined_context)
            if "LLM отключён" in fallback and "Найдены" not in fallback:
                return (
                    build_empty_context_message(
                        mode="idea_evaluation",
                        llm_enabled=self._settings.llm_enabled,
                        llm_available=False,
                        attestation=build_search_attestation([], local_chunks=0, external_papers=0),
                    ),
                    None,
                    "llm=disabled",
                )
            return fallback, None, "llm=disabled"

        system_prompt = get_agent_idea_eval_prompt(self._settings)
        raw = self._generate_with_prompt(message, combined_context, system_prompt)
        payload, detail = self._parse_idea_payload(raw)

        if payload is None:
            retry_raw = self._generate_with_prompt(
                f"{message}\n\nВерни только валидный JSON без markdown.",
                combined_context,
                system_prompt,
            )
            payload, retry_detail = self._parse_idea_payload(retry_raw)
            detail = f"retry=parse;{retry_detail or detail}"

        if payload is None:
            raw_text = RagService._dedupe_answer(raw).strip()
            summary = raw_text[:2000] if raw_text else "Не удалось сформировать оценку идеи."
            fallback_assessment = IdeaAssessment(
                relevance=RelevanceAssessment(
                    level="low",
                    criteria=[],
                    rationale="Структурированная оценка недоступна.",
                ),
                evidence_for=[],
                evidence_against=[],
                success_outlook="",
                confidence="low",
            )
            return summary, fallback_assessment, detail or "parse=failed"

        assessment = self._build_idea_assessment(payload)
        if external_papers and not self._assessment_has_external_citations(assessment):
            logger.warning(
                "Idea assessment missing external citations",
                extra={"event": "agent.idea.missing_external_citations"},
            )
            retry_prompt = (
                f"{system_prompt}\n\n"
                "В предыдущем JSON не было ссылок [En] на внешние публикации. "
                "Перепиши JSON: relevance, evidence и summary должны содержать [En], "
                "если во внешнем контексте есть публикации."
            )
            retry_raw = self._generate_with_prompt(message, combined_context, retry_prompt)
            retry_payload, retry_detail = self._parse_idea_payload(retry_raw)
            if retry_payload is not None:
                retry_assessment = self._build_idea_assessment(retry_payload)
                if self._assessment_has_external_citations(retry_assessment):
                    assessment = retry_assessment
                    payload = retry_payload
                    detail = f"retry=external;{retry_detail or detail}"
                else:
                    detail = f"retry=external_failed;{retry_detail or detail}"
            else:
                detail = f"retry=external_failed;{retry_detail or detail}"

        if not self._assessment_evidence_is_substantive(assessment):
            logger.warning(
                "Idea assessment evidence is shallow or missing citations",
                extra={"event": "agent.idea.shallow_evidence"},
            )
            retry_prompt = f"{system_prompt}\n\n{_IDEA_EVIDENCE_RETRY_SUFFIX}"
            retry_raw = self._generate_with_prompt(message, combined_context, retry_prompt)
            retry_payload, retry_detail = self._parse_idea_payload(retry_raw)
            if retry_payload is not None:
                retry_assessment = self._build_idea_assessment(retry_payload)
                if self._assessment_evidence_is_substantive(retry_assessment):
                    assessment = retry_assessment
                    payload = retry_payload
                    detail = f"retry=evidence;{retry_detail or detail}"
                else:
                    detail = f"retry=evidence_failed;{retry_detail or detail}"
            else:
                detail = f"retry=evidence_failed;{retry_detail or detail}"

        answer = payload.summary.strip() or self._format_idea_answer(assessment)
        answer = RagService._dedupe_answer(answer)
        return answer, assessment, detail or "parse=ok"

    def _parse_idea_payload(self, raw: str) -> tuple[_IdeaEvalPayload | None, str | None]:
        text = raw.strip()
        if not text:
            return None, "empty"

        candidates = [text]
        match = _JSON_BLOCK_PATTERN.search(text)
        if match:
            candidates.insert(0, match.group(0))

        for candidate in candidates:
            try:
                data = json.loads(candidate)
                payload = _IdeaEvalPayload.model_validate(data)
                return payload, "json=ok"
            except (json.JSONDecodeError, ValidationError):
                continue

        return None, "json=invalid"

    @staticmethod
    def _evidence_has_citation(text: str) -> bool:
        return bool(_CITATION_PATTERN.search(text))

    @classmethod
    def _assessment_evidence_has_citations(cls, assessment: IdeaAssessment) -> bool:
        evidence_items = [*assessment.evidence_for, *assessment.evidence_against]
        if not evidence_items:
            return True
        return all(cls._evidence_has_citation(item.text) for item in evidence_items)

    @classmethod
    def _assessment_evidence_is_substantive(cls, assessment: IdeaAssessment) -> bool:
        evidence_items = [*assessment.evidence_for, *assessment.evidence_against]
        if not evidence_items:
            return True
        return all(
            cls._evidence_has_citation(item.text) and not cls._is_shallow_evidence(item.text)
            for item in evidence_items
        )

    @classmethod
    def _is_shallow_evidence(cls, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        without_citations = _CITATION_PATTERN.sub("", stripped).strip()
        if len(without_citations) < _MIN_EVIDENCE_SUBSTANCE_LENGTH:
            return True
        lowered = stripped.lower()
        return any(pattern in lowered for pattern in _SHALLOW_META_PATTERNS)

    @classmethod
    def _parse_relevance(cls, raw: Any) -> RelevanceAssessment:
        if isinstance(raw, RelevanceAssessment):
            return raw
        if isinstance(raw, dict):
            return RelevanceAssessment.model_validate(raw)
        if isinstance(raw, str):
            text = raw.strip()
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
        raise ValueError("invalid relevance payload")

    @classmethod
    def _assessment_has_external_citations(cls, assessment: IdeaAssessment) -> bool:
        texts = [
            assessment.relevance.rationale,
            *[criterion.detail for criterion in assessment.relevance.criteria],
            *[item.text for item in assessment.evidence_for],
            *[item.text for item in assessment.evidence_against],
        ]
        return any(_EXTERNAL_CITATION_PATTERN.search(text) for text in texts)

    @staticmethod
    def _build_idea_assessment(payload: _IdeaEvalPayload) -> IdeaAssessment:
        confidence = payload.confidence.strip().casefold()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"

        if payload.relevance is not None:
            relevance = ResearchAgent._parse_relevance(payload.relevance)
        else:
            level = (payload.relevance_level or "medium").strip().casefold()
            if level not in {"low", "medium", "high"}:
                level = "medium"
            rationale = (payload.relevance_rationale or "").strip()
            relevance = RelevanceAssessment(
                level=level,  # type: ignore[arg-type]
                criteria=[
                    RelevanceCriterion(
                        name="topic_overlap",
                        level=level,  # type: ignore[arg-type]
                        detail=rationale,
                    )
                ] if rationale else [],
                rationale=rationale,
            )

        return IdeaAssessment(
            relevance=relevance,
            evidence_for=[EvidenceItem(text=item.strip()) for item in payload.evidence_for if item.strip()],
            evidence_against=[
                EvidenceItem(text=item.strip()) for item in payload.evidence_against if item.strip()
            ],
            success_outlook=payload.success_outlook.strip(),
            confidence=confidence,  # type: ignore[arg-type]
        )

    @staticmethod
    def _format_idea_answer(assessment: IdeaAssessment) -> str:
        level_labels = {"low": "низкая", "medium": "средняя", "high": "высокая"}
        criterion_labels = {
            "local_sources": "Локальные источники",
            "external_publications": "Внешние публикации",
            "topic_overlap": "Пересечение с темой",
        }
        lines = [
            f"Релевантность: {level_labels.get(assessment.relevance.level, assessment.relevance.level)}",
            "",
            "Критерии:",
        ]
        for criterion in assessment.relevance.criteria:
            name = criterion_labels.get(criterion.name, criterion.name)
            level = level_labels.get(criterion.level, criterion.level)
            lines.append(f"• {name} ({level}): {criterion.detail}")
        if assessment.relevance.rationale.strip():
            lines.extend(["", f"Обоснование: {assessment.relevance.rationale}", ""])
        lines.append("Аргументы за:")
        lines.extend(f"• {item.text}" for item in assessment.evidence_for)
        lines.extend(["", "Аргументы против:"])
        lines.extend(f"• {item.text}" for item in assessment.evidence_against)
        lines.extend(
            [
                "",
                f"Перспективы: {assessment.success_outlook}",
                f"Уверенность: {level_labels.get(assessment.confidence, assessment.confidence)}",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _is_shallow_answer(answer: str) -> bool:
        stripped = answer.strip()
        if not stripped:
            return True
        if len(stripped) < 200:
            return True
        if not _CITATION_PATTERN.search(stripped):
            return True
        lowered = stripped.lower()
        return any(pattern in lowered for pattern in _SHALLOW_META_PATTERNS)

    def _generate_with_prompt(
        self,
        question: str,
        context: str,
        system_prompt: str,
    ) -> str:
        provider = self._llm
        if provider is None:
            return ""

        if hasattr(provider, "_system_prompt"):
            original = provider._system_prompt
            provider._system_prompt = system_prompt
            try:
                return provider.generate(question, context)
            finally:
                provider._system_prompt = original

        return provider.generate(question, context)

    @staticmethod
    def _fallback_answer(local_context: str, combined_context: str) -> str:
        if local_context.strip():
            preview = local_context.strip().split("\n\n", 1)[0]
            return f"{_LLM_DISABLED_MESSAGE}\n\n{preview}"
        if combined_context.strip():
            preview = combined_context.strip().split("\n\n", 1)[0]
            return f"{_LLM_DISABLED_MESSAGE}\n\n{preview}"
        return build_empty_context_message(
            mode="question",
            llm_enabled=False,
            llm_available=False,
            attestation=build_search_attestation([], local_chunks=0, external_papers=0),
        )


def create_research_agent(
    settings: Settings,
    hybrid_search: HybridSearcher,
    rag_service: RagService,
    literature_service: ExternalLiteratureService,
    llm_provider: LLMProvider | None = None,
) -> ResearchAgent:
    return ResearchAgent(
        hybrid_search=hybrid_search,
        rag_service=rag_service,
        literature_service=literature_service,
        llm_provider=llm_provider,
        settings=settings,
    )
