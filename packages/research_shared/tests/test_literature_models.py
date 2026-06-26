
from research_shared.literature import ExternalPaper, LiteratureSearchProvider

from research_shared.literature.models import ExternalPaper as ExternalPaperModel


def test_external_paper_required_fields() -> None:
    paper = ExternalPaper(
        title="Test",
        url="https://example.org",
        source="openalex",
    )
    assert paper.abstract == ""
    assert paper.authors == []
    assert paper.doi is None


def test_literature_package_exports() -> None:
    assert ExternalPaper is ExternalPaperModel


def test_settings_literature_defaults() -> None:
    from research_shared.config.settings import Settings

    settings = Settings(_env_file=None)
    assert settings.literature_cache_ttl_seconds == 3600
    assert settings.literature_default_limit == 10
    assert settings.semantic_scholar_api_key == ""
