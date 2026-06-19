from __future__ import annotations

import time

from research_shared.logging_config import get_logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
  async def dispatch(self, request: Request, call_next) -> Response:
    started = time.perf_counter()
    logger.info(
      "HTTP request started",
      extra={
        "method": request.method,
        "path": request.url.path,
        "event": "http.request.start",
      },
    )
    try:
      response = await call_next(request)
    except Exception:
      logger.exception(
        "HTTP request failed",
        extra={
          "method": request.method,
          "path": request.url.path,
          "event": "http.request.error",
        },
      )
      raise

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info(
      "HTTP request completed",
      extra={
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "count": duration_ms,
        "event": "http.request.complete",
      },
    )
    return response
