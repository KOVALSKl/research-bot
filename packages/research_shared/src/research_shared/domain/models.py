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
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchQuery(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    search_type: SearchType = SearchType.HYBRID


class SearchResult(BaseModel):
    chunk: ResearchChunk
    score: float
    search_type: SearchType = SearchType.HYBRID
