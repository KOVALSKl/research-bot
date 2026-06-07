from unittest.mock import MagicMock, patch

from research_shared.config.settings import Settings
from research_shared.storage.embeddings.factory import create_dense_embedder, probe_dense_vector_size
from research_shared.storage.embeddings.ollama import OllamaDenseEmbedder


def test_settings_ollama_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.dense_embedding_provider == "ollama"
    assert settings.ollama_embedding_model == "qwen3-embedding:0.6b"
    assert settings.ollama_url == "http://localhost:11434"
    assert settings.dense_vector_size == 1024


def test_create_dense_embedder_ollama() -> None:
    settings = Settings(_env_file=None, dense_embedding_provider="ollama")
    embedder = create_dense_embedder(settings)
    assert isinstance(embedder, OllamaDenseEmbedder)


def test_create_dense_embedder_fastembed() -> None:
    from research_shared.storage.embeddings.fastembed import FastEmbedDenseEmbedder

    settings = Settings(_env_file=None, dense_embedding_provider="fastembed")
    embedder = create_dense_embedder(settings)
    assert isinstance(embedder, FastEmbedDenseEmbedder)


def test_ollama_dense_embedder_batch() -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2], [0.3, 0.4]],
    }

    mock_http = MagicMock()
    mock_http.post.return_value = mock_response

    embedder = OllamaDenseEmbedder(
        model="qwen3-embedding:0.6b",
        base_url="http://localhost:11434",
    )

    with patch("httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value = mock_http
        result = embedder.embed(["hello", "world"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2]
    mock_http.post.assert_called_once_with(
        "http://localhost:11434/api/embed",
        json={"model": "qwen3-embedding:0.6b", "input": ["hello", "world"]},
    )


def test_probe_dense_vector_size() -> None:
    settings = Settings(_env_file=None, dense_embedding_provider="ollama")

    with patch(
        "research_shared.storage.embeddings.factory.create_dense_embedder"
    ) as mock_create:
        mock_embedder = MagicMock()
        mock_embedder.probe_vector_size.return_value = 1024
        mock_create.return_value = mock_embedder
        assert probe_dense_vector_size(settings) == 1024
