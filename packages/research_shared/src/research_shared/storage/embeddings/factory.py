from research_shared.config.settings import Settings
from research_shared.storage.embeddings.fastembed import FastEmbedDenseEmbedder, FastEmbedSparseEncoder
from research_shared.storage.embeddings.ollama import OllamaDenseEmbedder
from research_shared.storage.protocols import DenseEmbedder, SparseEncoder


def create_dense_embedder(settings: Settings | None = None) -> DenseEmbedder:
    settings = settings or Settings()

    if settings.dense_embedding_provider == "ollama":
        return OllamaDenseEmbedder(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_url,
            timeout_seconds=settings.ollama_timeout_seconds,
        )

    return FastEmbedDenseEmbedder(settings.dense_embedding_model)


def create_sparse_encoder(settings: Settings | None = None) -> SparseEncoder:
    settings = settings or Settings()
    return FastEmbedSparseEncoder(settings.sparse_embedding_model)


def probe_dense_vector_size(settings: Settings | None = None) -> int:
    settings = settings or Settings()
    embedder = create_dense_embedder(settings)
    if hasattr(embedder, "probe_vector_size"):
        return embedder.probe_vector_size()
    return len(embedder.embed(["probe"])[0])
