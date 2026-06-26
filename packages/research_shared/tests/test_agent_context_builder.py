from research_shared.agents.context_builder import build_agent_context


def test_build_agent_context_local_only() -> None:
    result = build_agent_context("[1] Local paper\nBody", "")
    assert result.startswith("Локальные источники:")
    assert "[1] Local paper" in result
    assert "Внешние публикации:" not in result


def test_build_agent_context_external_only() -> None:
    result = build_agent_context("", "[E1] External\nAbstract")
    assert result.startswith("Внешние публикации:")
    assert "[E1] External" in result
    assert "Локальные источники:" not in result


def test_build_agent_context_combined() -> None:
    result = build_agent_context("[1] Local", "[E1] External")
    assert "Локальные источники:" in result
    assert "Внешние публикации:" in result
    assert "[1] Local" in result
    assert "[E1] External" in result
