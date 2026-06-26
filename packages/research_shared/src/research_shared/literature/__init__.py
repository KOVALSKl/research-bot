from research_shared.literature.models import ExternalPaper, LiteratureSource
from research_shared.literature.protocols import LiteratureSearchProvider
from research_shared.literature.service import ExternalLiteratureService, create_literature_service

__all__ = [
    "ExternalLiteratureService",
    "ExternalPaper",
    "LiteratureSearchProvider",
    "LiteratureSource",
    "create_literature_service",
]
