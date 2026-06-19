from __future__ import annotations


def is_outgoing(*, from_me: bool = False, out: int = 0, from_id: int = 0) -> bool:
  """Return True if the VK message was sent by the bot or the community."""
  return from_me or out == 1 or from_id < 0
