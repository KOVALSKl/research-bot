from typing import Literal

from pydantic import BaseModel, Field

LiteratureSource = Literal["openalex", "arxiv", "semantic_scholar"]


class ExternalPaper(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    doi: str | None = None
    url: str
    pdf_url: str | None = None
    source: LiteratureSource | str
