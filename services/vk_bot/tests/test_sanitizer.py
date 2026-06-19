import pytest

from vk_bot.config import VkBotSettings
from vk_bot.security.rate_limiter import MemoryRateLimiter
from vk_bot.security.sanitizer import MessageSanitizer


@pytest.fixture
def settings() -> VkBotSettings:
  return VkBotSettings(
    vk_max_question_length=100,
    vk_debounce_seconds=2,
    vk_rate_limit_max_messages=2,
    vk_rate_limit_window_seconds=60,
  )


def test_sanitizer_strips_control_chars(settings):
  sanitizer = MessageSanitizer(settings)
  result = sanitizer.sanitize("  hello\x00world  ")
  assert result == "helloworld"


def test_sanitizer_truncates(settings):
  sanitizer = MessageSanitizer(settings)
  result = sanitizer.sanitize("a" * 200)
  assert len(result) == 100


@pytest.mark.asyncio
async def test_rate_limiter_blocks_after_max(settings):
  limiter = MemoryRateLimiter(settings)
  assert await limiter.allow(1, "msg1")
  assert await limiter.allow(1, "msg2")
  assert not await limiter.allow(1, "msg3")


@pytest.mark.asyncio
async def test_debounce_ignores_duplicate(settings):
  limiter = MemoryRateLimiter(settings)
  text = "same question text here"
  assert await limiter.allow(1, text)
  assert not await limiter.allow(1, text)
