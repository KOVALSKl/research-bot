from research_shared.domain.models import Citation
from research_shared.rag.citations import group_citations_by_document


def _citation(
    *,
    research_id: str = "r1",
    display_name: str = "Paper A",
    page: int | None = 1,
    score: float = 0.8,
    authors: list[str] | None = None,
) -> Citation:
    return Citation(
        research_id=research_id,
        title=display_name,
        page=page,
        score=score,
        display_name=display_name,
        authors=authors or ["Alice"],
    )


def test_group_same_document_multiple_pages() -> None:
    citations = [
        _citation(page=5, score=0.9),
        _citation(page=12, score=0.7),
        _citation(page=23, score=0.85),
    ]

    groups = group_citations_by_document(citations)

    assert len(groups) == 1
    assert groups[0].pages == [5, 12, 23]
    assert groups[0].max_score == 0.9


def test_group_different_documents() -> None:
    citations = [
        _citation(research_id="r1", display_name="Paper A", page=1),
        _citation(research_id="r2", display_name="Paper B", page=2),
    ]

    groups = group_citations_by_document(citations)

    assert len(groups) == 2
    assert groups[0].display_name == "Paper A"
    assert groups[1].display_name == "Paper B"


def test_group_preserves_first_appearance_order() -> None:
    citations = [
        _citation(research_id="r2", display_name="Second", page=1),
        _citation(research_id="r1", display_name="First", page=2),
        _citation(research_id="r2", display_name="Second", page=3),
    ]

    groups = group_citations_by_document(citations)

    assert [group.display_name for group in groups] == ["Second", "First"]
    assert groups[0].pages == [1, 3]


def test_group_deduplicates_pages() -> None:
    citations = [
        _citation(page=5, score=0.6),
        _citation(page=5, score=0.9),
    ]

    groups = group_citations_by_document(citations)

    assert groups[0].pages == [5]
    assert groups[0].max_score == 0.9


def test_group_citation_without_page() -> None:
    citations = [_citation(page=None)]

    groups = group_citations_by_document(citations)

    assert groups[0].pages == []
