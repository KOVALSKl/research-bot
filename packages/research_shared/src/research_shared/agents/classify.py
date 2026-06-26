from __future__ import annotations

import re

from research_shared.agents.models import AgentAskRequest, ResolvedAgentMode
from research_shared.config.settings import Settings
from research_shared.llm.protocols import LLMProvider
from research_shared.logging_config import get_logger

logger = get_logger(__name__)

_IDEA_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bоцени\s+(?:мою\s+)?идею\b", re.IGNORECASE),
    re.compile(r"\bнасколько\s+перспектив", re.IGNORECASE),
    re.compile(r"\b(?:насколько|насколько)\s+актуальн", re.IGNORECASE),
    re.compile(r"\bпредлагаю\s+(?:исследовать|изучить|разработать)\b", re.IGNORECASE),
    re.compile(r"\b(?:research|scientific)\s+idea\b", re.IGNORECASE),
    re.compile(r"\bevaluate\s+(?:my\s+)?idea\b", re.IGNORECASE),
    re.compile(r"\bhow\s+promising\b", re.IGNORECASE),
    re.compile(r"\bidea\s+evaluation\b", re.IGNORECASE),
    re.compile(r"\bгипотез[аы]\b.*\b(?:перспектив|оцен)", re.IGNORECASE),
)


def classify_by_rules(message: str) -> ResolvedAgentMode | None:
    text = message.strip()
    if not text:
        return None
    for pattern in _IDEA_PATTERNS:
        if pattern.search(text):
            return "idea_evaluation"
    return None


def classify_request(
    request: AgentAskRequest,
    *,
    settings: Settings | None = None,
    llm: LLMProvider | None = None,
) -> ResolvedAgentMode:
    if request.mode == "question":
        return "question"
    if request.mode == "idea_evaluation":
        return "idea_evaluation"

    rule_result = classify_by_rules(request.message)
    if rule_result is not None:
        return rule_result

    if llm is not None and settings is not None and settings.llm_enabled:
        try:
            llm_result = _classify_with_llm(request.message, llm)
            if llm_result is not None:
                return llm_result
        except Exception:
            logger.exception(
                "LLM classification failed; defaulting to question",
                extra={"event": "agent.classify.llm.error"},
            )

    return "question"


def _classify_with_llm(message: str, llm: LLMProvider) -> ResolvedAgentMode | None:
    prompt = (
        "Classify the user message as either question or idea_evaluation.\n"
        "idea_evaluation = user describes a research idea/hypothesis and wants assessment.\n"
        "question = user asks for factual information or explanation.\n"
        "Reply with exactly one word: question or idea_evaluation."
    )
    original = getattr(llm, "_system_prompt", None)
    if hasattr(llm, "_system_prompt"):
        llm._system_prompt = prompt
    try:
        raw = llm.generate(message, "").strip().casefold()
    finally:
        if hasattr(llm, "_system_prompt") and original is not None:
            llm._system_prompt = original

    if "idea" in raw:
        return "idea_evaluation"
    if "question" in raw:
        return "question"
    return None
