from __future__ import annotations

from vk_bot.core_api.client import CoreApiError
from vk_bot.handlers import messages as ux

_NETWORK_MARKERS = (
  "connecterror",
  "connect timeout",
  "connection refused",
  "remoteprotocolerror",
  "connection reset",
  "connection error",
)


def map_core_api_error(exc: CoreApiError) -> str:
  message = str(exc)
  lower = message.lower()

  if "timed out" in lower or "timeout" in lower:
    return ux.ask_timeout()

  if exc.status_code == 503:
    return ux.service_starting()

  if exc.status_code is not None and exc.status_code >= 500:
    return ux.service_unavailable()

  if exc.status_code is None or any(marker in lower for marker in _NETWORK_MARKERS):
    return ux.connection_error()

  return ux.service_unavailable()
