import pytest

from research_shared.logging_config import _LOG_RECORD_RESERVED, log_extra


@pytest.mark.parametrize(
    "field",
    sorted(_LOG_RECORD_RESERVED),
)
def test_log_extra_sanitizes_reserved_logrecord_fields(field: str) -> None:
    extra = log_extra(**{field: "value", "event": "test"})
    assert field not in extra
    assert extra[f"ctx_{field}"] == "value"
    assert extra["event"] == "test"


def test_get_logger_sanitizes_filename_in_extra() -> None:
    from research_shared.logging_config import get_logger

    logger = get_logger("test.safe_logger")
    try:
        logger.info("upload", extra={"filename": "paper.pdf", "event": "test"})
    except KeyError as exc:
        pytest.fail(f"get_logger must avoid LogRecord conflicts: {exc}")
