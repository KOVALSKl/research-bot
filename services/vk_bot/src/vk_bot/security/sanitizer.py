from __future__ import annotations

import re

from research_shared.logging_config import get_logger
from vk_bot.config import VkBotSettings

logger = get_logger(__name__)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Model-specific control tokens that have no place in user messages
_CONTROL_TOKENS = re.compile(
    r"<\|system\|>|<\|user\|>|<\|assistant\|>|\[/?INST\]|\[/?SYS\]",
    re.IGNORECASE,
)

# Injection phrases replaced with a neutral marker so the message still
# reaches the LLM (avoiding false-positive blocks on legitimate text)
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)\bignore\s+(all\s+)?previous\b"), "[filtered]"),
    (re.compile(r"(?i)\bsystem\s*:"), "[filtered]"),
    (re.compile(r"(?i)\byou\s+are\s+now\b"), "[filtered]"),
    (re.compile(r"(?i)\bdisregard\b"), "[filtered]"),
]


class MessageSanitizer:
    def __init__(self, settings: VkBotSettings) -> None:
        self._max_length = settings.vk_max_question_length

    def sanitize(self, text: str) -> str:
        cleaned = text.replace("\x00", "")
        cleaned = _CONTROL_CHARS.sub("", cleaned)
        cleaned = _CONTROL_TOKENS.sub("", cleaned)
        cleaned = cleaned.strip()

        for pattern, replacement in _INJECTION_PATTERNS:
            if pattern.search(cleaned):
                logger.warning(
                    "Prompt-injection pattern detected and replaced",
                    extra={"event": "security.injection_replaced", "pattern": pattern.pattern},
                )
                cleaned = pattern.sub(replacement, cleaned)

        if len(cleaned) > self._max_length:
            cleaned = cleaned[: self._max_length]
        return cleaned
