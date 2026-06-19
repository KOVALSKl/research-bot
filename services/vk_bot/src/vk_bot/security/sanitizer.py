from __future__ import annotations

import logging
import re

from vk_bot.config import VkBotSettings

logger = logging.getLogger(__name__)

_INJECTION_MARKERS = (
  "system:",
  "ignore previous",
  "ignore all previous",
  "you are now",
  "disregard",
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class MessageSanitizer:
  def __init__(self, settings: VkBotSettings) -> None:
    self._max_length = settings.vk_max_question_length

  def sanitize(self, text: str) -> str:
    cleaned = text.replace("\x00", "")
    cleaned = _CONTROL_CHARS.sub("", cleaned)
    cleaned = cleaned.strip()

    lowered = cleaned.lower()
    for marker in _INJECTION_MARKERS:
      if marker in lowered:
        logger.warning("Potential prompt-injection marker detected: %s", marker)

    if len(cleaned) > self._max_length:
      cleaned = cleaned[: self._max_length]
    return cleaned
