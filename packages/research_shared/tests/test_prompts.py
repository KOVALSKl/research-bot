import pytest
from pydantic import ValidationError

from research_shared.config.settings import Settings
from research_shared.llm.prompts import DEFAULT_RAG_SYSTEM_PROMPT, get_rag_system_prompt


def test_get_rag_system_prompt_returns_default_when_empty() -> None:
    settings = Settings(_env_file=None, rag_system_prompt="")
    assert get_rag_system_prompt(settings) == DEFAULT_RAG_SYSTEM_PROMPT


def test_get_rag_system_prompt_returns_custom_override() -> None:
    custom = "Custom RAG instructions."
    settings = Settings(_env_file=None, rag_system_prompt=custom)
    assert get_rag_system_prompt(settings) == custom


def test_get_rag_system_prompt_strips_whitespace_only_override() -> None:
    settings = Settings(_env_file=None, rag_system_prompt="   ")
    assert get_rag_system_prompt(settings) == DEFAULT_RAG_SYSTEM_PROMPT


def test_default_rag_system_prompt_contains_v9_instructions() -> None:
    prompt = DEFAULT_RAG_SYSTEM_PROMPT
    assert "Структура ответа" in prompt
    assert "[1]" in prompt
    assert "одним связным текстом" not in prompt
    assert "подзаголовками" in prompt or "нумерованными" in prompt


def test_settings_ask_default_limit_defaults_and_env() -> None:
    settings = Settings(_env_file=None)
    assert settings.ask_default_limit == 10

    overridden = Settings(_env_file=None, ask_default_limit=15)
    assert overridden.ask_default_limit == 15


def test_settings_ask_default_limit_validation() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ask_default_limit=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ask_default_limit=51)
