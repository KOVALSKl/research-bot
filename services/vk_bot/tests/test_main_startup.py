import logging
import os

import httpx
import pytest
from httpx import MockTransport, Request, Response

from vk_bot.config import VkBotSettings
from vk_bot.main import logger, wait_for_core_api


def test_startup_log_emits_git_sha_and_redirect_handler(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.INFO, logger="vk_bot.main")
    monkeypatch.setenv("GIT_SHA", "abc123")

    logger.info(
        "Starting VK bot",
        extra={
            "event": "vk_bot.start",
            "git_sha": os.environ.get("GIT_SHA", "unknown"),
            "redirect_handler": "shared",
            "storage_backend": "yandex",
        },
    )

    records = [record for record in caplog.records if record.getMessage() == "Starting VK bot"]
    assert records
    assert getattr(records[0], "git_sha", None) == "abc123"
    assert getattr(records[0], "redirect_handler", None) == "shared"
    assert getattr(records[0], "storage_backend", None) == "yandex"


@pytest.mark.asyncio
async def test_wait_for_core_api_ready_after_failed_attempts() -> None:
    settings = VkBotSettings(
        core_api_base_url="http://testserver",
        vk_core_api_startup_wait_seconds=10,
        vk_core_api_startup_poll_interval_seconds=0.01,
    )
    attempts = {"count": 0}

    def handler(request: Request) -> Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise httpx.ConnectError("connection refused")
        return Response(200, json={"status": "ok"})

    transport = MockTransport(handler)
    client = httpx.AsyncClient(base_url=settings.core_api_base_url, transport=transport)

    await wait_for_core_api(settings, client=client)
    assert attempts["count"] == 3
    await client.aclose()


@pytest.mark.asyncio
async def test_wait_for_core_api_timeout_exits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = VkBotSettings(
        core_api_base_url="http://testserver",
        vk_core_api_startup_wait_seconds=0.05,
        vk_core_api_startup_poll_interval_seconds=0.01,
    )
    caplog.set_level(logging.ERROR, logger="vk_bot.main")

    def handler(_request: Request) -> Response:
        raise httpx.ConnectError("connection refused")

    transport = MockTransport(handler)
    client = httpx.AsyncClient(base_url=settings.core_api_base_url, transport=transport)

    with pytest.raises(SystemExit) as exc_info:
        await wait_for_core_api(settings, client=client)
    assert exc_info.value.code == 1

    timeout_records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "vk_bot.core_api_wait_timeout"
    ]
    assert timeout_records
    await client.aclose()
