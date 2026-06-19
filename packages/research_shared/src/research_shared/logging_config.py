from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# Attributes defined on logging.LogRecord — must not be passed via ``extra=``.
_LOG_RECORD_RESERVED = frozenset(
  logging.makeLogRecord({}).__dict__.keys()
) | frozenset({"message", "asctime"})


def log_extra(**fields: Any) -> dict[str, Any]:
  """Return an ``extra`` dict safe for ``logger.*(..., extra=...)``."""
  extra: dict[str, Any] = {}
  for key, value in fields.items():
    safe_key = f"ctx_{key}" if key in _LOG_RECORD_RESERVED else key
    extra[safe_key] = value
  return extra


class SafeLoggerAdapter(logging.LoggerAdapter):
  """Logger adapter that sanitizes ``extra`` fields before each log call."""

  def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    extra = kwargs.get("extra")
    if isinstance(extra, dict):
      kwargs["extra"] = log_extra(**extra)
    return msg, kwargs


def get_logger(name: str) -> SafeLoggerAdapter:
  return SafeLoggerAdapter(logging.getLogger(name), {})


class JsonLogFormatter(logging.Formatter):
  def format(self, record: logging.LogRecord) -> str:
    payload: dict[str, Any] = {
      "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
      "level": record.levelname,
      "logger": record.name,
      "message": record.getMessage(),
    }
    for key, value in record.__dict__.items():
      if key in _LOG_RECORD_RESERVED or key.startswith("_"):
        continue
      payload[key] = value
    if record.exc_info:
      payload["exception"] = self.formatException(record.exc_info)
    return json.dumps(payload, ensure_ascii=False)


def configure_logging(*, level: str = "INFO", log_format: str = "text") -> None:
  root = logging.getLogger()
  root.handlers.clear()
  root.setLevel(level.upper())

  handler = logging.StreamHandler(sys.stdout)
  if log_format.lower() == "json":
    handler.setFormatter(JsonLogFormatter())
  else:
    handler.setFormatter(
      logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
    )
  root.addHandler(handler)
