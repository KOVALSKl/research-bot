from __future__ import annotations

from research_shared.logging_config import configure_logging
from research_shared.config.settings import Settings


def setup_logging(settings: Settings) -> None:
  configure_logging(level=settings.log_level, log_format=settings.log_format)
