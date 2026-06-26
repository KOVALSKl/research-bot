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

    dense_embedding_provider: Literal["ollama", "fastembed"] = "fastembed"
    dense_vector_size: int = 1024

    ollama_host: str = "localhost"
    ollama_port: int = 11434
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    ollama_timeout_seconds: float = 120.0

    dense_embedding_model: str = "intfloat/multilingual-e5-large"
    sparse_embedding_model: str = "Qdrant/bm25"

    core_api_host: str = "0.0.0.0"
    core_api_port: int = 8000

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # Document storage (v10): local filesystem or Yandex Disk
    storage_backend: Literal["local", "yandex"] = "local"
    yandex_disk_api_token: str = ""
    yandex_disk_base_path: str = "disk:/research-docs"

    # Ingestion
    researches_dir: str = "researches"
    ingest_staging_dir: str = "researches/.staging"
    ingest_staging_ttl_hours: float = 24.0
    ingest_sync: bool = False
    ingest_state_collection: str = "ingestion_state"
    chunk_size: int = 900
    chunk_overlap: int = 180
    chunk_min_chars: int = Field(default=80, ge=0)

    # Background scanner (target 4) — OFF by default
    researches_scan_enabled: bool = False
    researches_scan_interval_seconds: float = 300.0

    # LLM (target 5) — OFF by default; default provider = huggingface when enabled
    llm_enabled: bool = False
    llm_provider: Literal["huggingface", "ollama", "custom", "yandex_ai_studio"] = "huggingface"
    llm_provider_module: str = ""
    hf_api_token: str = ""
    hf_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    hf_timeout_seconds: float = 120.0
    ollama_chat_model: str = "qwen3:0.6b"

    # Yandex AI Studio (openai SDK + Responses API)
    yandex_ai_studio_api_key: str = ""
    yandex_ai_studio_folder_id: str = ""
    yandex_ai_studio_model: str = "qwen3-235b-a22b-fp8/latest"
    yandex_ai_studio_base_url: str = "https://ai.api.cloud.yandex.net/v1"
    yandex_ai_studio_temperature: float = 0.3
    yandex_ai_studio_max_output_tokens: int = 2000
    yandex_ai_studio_timeout_seconds: float = 120.0

    # RAG (/ask)
    ask_default_limit: int = Field(default=10, ge=1, le=50)
    rag_system_prompt: str = ""

    # Research Agent (v11)
    min_local_results: int = Field(default=3, ge=0)
    min_local_score: float = Field(default=0.5, ge=0.0, le=1.0)
    agent_use_llm_context_check: bool = False
    agent_question_prompt: str = ""

    # Research Agent v14 — relevance filter and query reformulation
    agent_relevance_filter_enabled: bool = True
    agent_min_chunk_score: float = Field(default=0.55, ge=0.0, le=1.0)
    agent_use_llm_relevance_filter: bool = False
    agent_relevance_prompt: str = ""
    agent_query_reformulation_enabled: bool = True
    agent_query_reformulation_prompt: str = ""
    agent_idea_query_reformulation_enabled: bool = True
    agent_idea_query_reformulation_prompt: str = ""
    agent_idea_min_external_papers: int = Field(default=1, ge=0)
    agent_idea_eval_prompt: str = ""

    # Research Agent v15 — multilingual retrieval and evidence synthesis
    agent_search_queries_max: int = Field(default=3, ge=1, le=5)
    agent_supplement_external: bool = False
    agent_synthesis_retry_on_shallow: bool = True
    agent_min_citations_per_section: int = Field(default=1, ge=0)

    # Research Agent v6 — ReAct loop
    agent_max_iterations: int = Field(default=6, ge=1, le=20)
    agent_react_system_prompt: str = ""
    agent_idea_min_local_chunks: int = Field(default=1, ge=0)

    # External literature search (OpenAlex, arXiv, Semantic Scholar)
    literature_cache_ttl_seconds: int = 3600
    literature_cache_empty_ttl_seconds: int = 60
    semantic_scholar_api_key: str = ""
    literature_default_limit: int = Field(default=10, ge=1, le=50)
    literature_idea_mode_limit: int = Field(default=15, ge=1, le=50)
    literature_idea_post_filter_min_score: float = Field(default=0.08, ge=0.0, le=1.0)

    # External PDF cache and delivery (v9)
    external_pdf_fetch_enabled: bool = True
    external_pdf_cache_dir: str = "external_pdfs"
    external_pdf_max_bytes: int = Field(default=52_428_800, ge=1)
    external_pdf_fetch_timeout_seconds: float = 60.0
    external_pdf_max_redirects: int = Field(default=10, ge=1, le=20)

    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    @model_validator(mode="after")
    def validate_llm_provider_config(self) -> Self:
        if self.llm_enabled and self.llm_provider == "custom" and not self.llm_provider_module.strip():
            raise ValueError(
                "llm_provider_module is required when llm_provider=custom and llm_enabled=true"
            )
        return self

    def effective_agent_use_llm_relevance_filter(self) -> bool:
        if not self.agent_relevance_filter_enabled:
            return False
        if self.agent_use_llm_relevance_filter:
            return True
        return self.llm_enabled

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
