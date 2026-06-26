from typing import Protocol, runtime_checkable

from research_shared.literature.models import ExternalPaper


@runtime_checkable
class LiteratureSearchProvider(Protocol):
    async def search(
        self,
        query: str,
        limit: int,
        year_from: int | None = None,
    ) -> list[ExternalPaper]: ...
