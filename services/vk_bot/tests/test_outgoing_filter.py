import pytest

from vk_bot.vk.outgoing_filter import is_outgoing


@pytest.mark.parametrize(
  ("from_me", "out", "from_id", "expected"),
  [
    (True, 0, 42, True),
    (False, 1, 42, True),
    (False, 0, -1, True),
    (False, 0, 42, False),
  ],
)
def test_is_outgoing(from_me: bool, out: int, from_id: int, expected: bool) -> None:
  assert is_outgoing(from_me=from_me, out=out, from_id=from_id) is expected
