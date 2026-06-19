import pytest

from vk_bot.config import VkBotSettings
from vk_bot.security.rate_limiter import MemoryRateLimiter


@pytest.fixture
def settings() -> VkBotSettings:
  return VkBotSettings(
    vk_rate_limit_backend="memory",
    vk_rate_limit_max_messages=10,
    vk_rate_limit_window_seconds=60,
    vk_debounce_seconds=2,
  )


@pytest.mark.asyncio
async def test_rate_limit_notify_cooldown(settings: VkBotSettings):
  limiter = MemoryRateLimiter(settings)

  assert await limiter.should_notify_rate_limit(1) is True
  await limiter.mark_rate_limit_notified(1)
  assert await limiter.should_notify_rate_limit(1) is False


@pytest.mark.asyncio
async def test_pdf_only_debounce(settings: VkBotSettings):
  limiter = MemoryRateLimiter(settings)

  assert await limiter.allow(1, "", attachment_count=1) is True
  assert await limiter.allow(1, "", attachment_count=1) is False
