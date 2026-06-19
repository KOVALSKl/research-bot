from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from vk_bot.domain import IncomingMessage


@runtime_checkable
class VkTransport(Protocol):
  """Receives VK events and dispatches normalized messages to a handler."""

  async def run(
    self,
    handler: Callable[[IncomingMessage], Awaitable[None]],
  ) -> None:
    """Start listening and invoke *handler* for each incoming message."""
    ...

  async def stop(self) -> None:
    """Stop the transport gracefully."""
    ...
