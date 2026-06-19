from unittest.mock import MagicMock, patch

from research_shared.config.settings import Settings
from research_shared.llm.factory import create_llm_provider
from research_shared.llm.huggingface import HuggingFaceInferenceProvider
from research_shared.llm.ollama import OllamaLLMProvider
from research_shared.llm.prompts import DEFAULT_RAG_SYSTEM_PROMPT, get_rag_system_prompt


def test_ollama_provider_sends_injected_system_prompt() -> None:
    custom_prompt = "Test system prompt."
    provider = OllamaLLMProvider(
        model="test-model",
        base_url="http://ollama:11434",
        system_prompt=custom_prompt,
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "answer"}}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("research_shared.llm.ollama.httpx.Client", return_value=mock_client):
        result = provider.generate("question?", "context text")

    assert result == "answer"
    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == custom_prompt
    assert "одним связным текстом" not in payload["messages"][0]["content"]


def test_huggingface_provider_sends_injected_system_prompt() -> None:
    custom_prompt = "HF test prompt."
    provider = HuggingFaceInferenceProvider(
        model="test-model",
        api_token="token",
        system_prompt=custom_prompt,
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "hf answer"}}],
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("research_shared.llm.huggingface.httpx.Client", return_value=mock_client):
        result = provider.generate("question?", "context text")

    assert result == "hf answer"
    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == custom_prompt
    assert "одним связным текстом" not in payload["messages"][0]["content"]


def test_create_llm_provider_uses_default_prompt_from_settings() -> None:
    settings = Settings(
        _env_file=None,
        llm_enabled=True,
        llm_provider="ollama",
        rag_system_prompt="",
    )
    expected_prompt = get_rag_system_prompt(settings)

    with patch(
        "research_shared.llm.factory.OllamaLLMProvider",
        autospec=True,
    ) as mock_cls:
        mock_cls.return_value = MagicMock()
        create_llm_provider(settings)

    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["system_prompt"] == expected_prompt
    assert mock_cls.call_args.kwargs["system_prompt"] == DEFAULT_RAG_SYSTEM_PROMPT


def test_create_llm_provider_uses_custom_prompt_override() -> None:
    custom = "Override prompt for RAG."
    settings = Settings(
        _env_file=None,
        llm_enabled=True,
        llm_provider="huggingface",
        hf_api_token="token",
        rag_system_prompt=custom,
    )

    with patch(
        "research_shared.llm.factory.HuggingFaceInferenceProvider",
        autospec=True,
    ) as mock_cls:
        mock_cls.return_value = MagicMock()
        create_llm_provider(settings)

    assert mock_cls.call_args.kwargs["system_prompt"] == custom


def test_create_llm_provider_returns_none_when_disabled() -> None:
    settings = Settings(_env_file=None, llm_enabled=False)
    assert create_llm_provider(settings) is None
