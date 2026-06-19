from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class SearchType(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class ResearchChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    research_id: str
    title: str
    text: str
    source_path: str | None = None
    display_name: str | None = None
    authors: list[str] = Field(default_factory=list)
    chapter: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchFilters(BaseModel):
    research_id: str | None = None
    source_path: str | None = None
    authors: list[str] | None = None
    page_min: int | None = None
    page_max: int | None = None
    chapter: str | None = None


class SearchQuery(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    search_type: SearchType = SearchType.HYBRID
    filters: SearchFilters | None = None


class SearchResult(BaseModel):
    chunk: ResearchChunk
    score: float
    search_type: SearchType = SearchType.HYBRID


class IngestStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    """Durable state of a source document tracked in the ingestion_state collection."""

    filename: str
    content_hash: str
    research_id: str
    display_name: str | None = None
    status: IngestStatus = IngestStatus.QUEUED
    chunk_count: int = 0
    indexed_at: datetime | None = None
    updated_at: datetime | None = None
    error: str | None = None


class DocumentListItem(BaseModel):
    research_id: str
    filename: str
    display_name: str | None = None
    status: IngestStatus
    chunk_count: int = 0
    indexed_at: datetime | None = None


class Citation(BaseModel):
    research_id: str
    title: str
    page: int | None = None
    score: float
    source_path: str | None = None
    filename: str | None = None
    display_name: str | None = None
    chapter: str | None = None
    authors: list[str] = Field(default_factory=list)


class SourceFileRef(BaseModel):
    research_id: str
    filename: str
    display_name: str | None = None
    path: str | None = None


class AskQuery(BaseModel):
    question: str
    limit: int | None = Field(default=None, ge=1, le=50)


class AskResponse(BaseModel):
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    context_chunks: list[ResearchChunk] = Field(default_factory=list)
    source_files: list[SourceFileRef] = Field(default_factory=list)
