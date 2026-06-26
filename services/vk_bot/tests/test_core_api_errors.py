from vk_bot.core_api.client import CoreApiError
from vk_bot.core_api.errors import map_core_api_error
from vk_bot.handlers import messages as ux


def test_map_core_api_error_connection() -> None:
  message = map_core_api_error(CoreApiError("Ask request failed: ConnectError('refused')"))
  assert message == ux.connection_error()
  assert "ConnectError" not in message
  assert "httpx" not in message.lower()


def test_map_core_api_error_service_unavailable() -> None:
  message = map_core_api_error(CoreApiError("internal", status_code=500))
  assert message == ux.service_unavailable()


def test_map_core_api_error_service_starting() -> None:
  message = map_core_api_error(CoreApiError("starting", status_code=503))
  assert message == ux.service_starting()


def test_map_core_api_error_timeout() -> None:
  message = map_core_api_error(CoreApiError("Agent stream request failed: timed out"))
  assert message == ux.ask_timeout()
