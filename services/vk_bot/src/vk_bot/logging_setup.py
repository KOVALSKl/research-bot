from __future__ import annotations

from research_shared.logging_config import configure_logging

from vk_bot.config import VkBotSettings


def setup_logging(settings: VkBotSettings) -> None:
  configure_logging(level=settings.log_level, log_format=settings.log_format)
