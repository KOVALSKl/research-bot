import pytest

from core_api.api.routes.health import health_check


@pytest.mark.asyncio
async def test_health_check() -> None:
    result = await health_check()
    assert result == {"status": "ok"}
