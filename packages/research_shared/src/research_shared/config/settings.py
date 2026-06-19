from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "research_chunks"
    qdrant_recreate_on_schema_mismatch: bool = True

    dense_embedding_provider: Literal["ollama", "fastembed"] = "ollama"
    dense_vector_size: int = 1024

    ollama_host: str = "localhost"
    ollama_port: int = 11434
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    ollama_timeout_seconds: float = 120.0

    dense_embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_embedding_model: str = "Qdrant/bm25"

    core_api_host: str = "0.0.0.0"
    core_api_port: int = 8000

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # Ingestion
    researches_dir: str = "researches"
    ingest_sync: bool = False
    ingest_state_collection: str = "ingestion_state"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Background scanner (target 4) — OFF by default
    researches_scan_enabled: bool = False
    researches_scan_interval_seconds: float = 300.0

    # LLM (target 5) — OFF by default; default provider = huggingface when enabled
    llm_enabled: bool = False
    llm_provider: Literal["huggingface", "ollama", "custom"] = "huggingface"
    llm_provider_module: str = ""
    hf_api_token: str = ""
    hf_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    hf_timeout_seconds: float = 120.0
    ollama_chat_model: str = "qwen3:0.6b"

    # RAG (/ask)
    ask_default_limit: int = Field(default=10, ge=1, le=50)
    rag_system_prompt: str = ""

    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    @model_validator(mode="after")
    def validate_llm_provider_config(self) -> Self:
        if self.llm_enabled and self.llm_provider == "custom" and not self.llm_provider_module.strip():
            raise ValueError(
                "llm_provider_module is required when llm_provider=custom and llm_enabled=true"
            )
        return self

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def ollama_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"

    @property
    def effective_celery_broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def effective_celery_result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
