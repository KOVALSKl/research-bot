from research_shared.domain.models import Citation
from research_shared.literature.models import ExternalPaper
from research_shared.rag.cited_sources import extract_cited_sources


def _local(index: int) -> Citation:
    return Citation(
        research_id=f"r{index}",
        title=f"Paper {index}",
        page=index,
        score=0.9,
        source_path=f"/data/researches/paper{index}.pdf",
        filename=f"paper{index}.pdf",
        authors=[f"Author {index}"],
    )


def _external(index: int) -> ExternalPaper:
    return ExternalPaper(
        title=f"External {index}",
        authors=[f"Ext {index}"],
        year=2020 + index,
        abstract="Abstract",
        doi=f"10.1000/{index}",
        url=f"https://example.org/{index}",
        source="openalex",
    )


def test_extract_mixed_local_and_external_indices() -> None:
    local = [_local(1), _local(2)]
    external = [
        ExternalPaper(
            title="External 1",
            authors=["Ext 1"],
            year=2021,
            abstract="Abstract",
            doi="10.1000/1",
            url="https://example.org/1",
            pdf_url="https://example.org/1.pdf",
            source="openalex",
        ),
        _external(2),
    ]
    answer = "Fact from [2] and external [E1] plus [1]."

    cited = extract_cited_sources(answer, local, external)

    assert [c.research_id for c in cited.local] == ["r2", "r1"]
    assert cited.local_indices == [2, 1]
    assert [p.title for p in cited.external] == ["External 1"]
    assert cited.external_indices == [1]
    assert len(cited.source_files) == 2
    assert len(cited.external_source_files) == 1
    assert cited.external_source_files[0].external_index == 1


def test_extract_out_of_range_markers_are_ignored() -> None:
    local = [_local(1)]
    external = [_external(1)]
    answer = "See [99] and [E99] and [1] and [E1]."

    cited = extract_cited_sources(answer, local, external)

    assert len(cited.local) == 1
    assert cited.local_indices == [1]
    assert len(cited.external) == 1
    assert cited.external_indices == [1]


def test_extract_fallback_when_no_markers() -> None:
    local = [_local(1), _local(2)]
    external = [_external(1)]
    answer = "Answer without citation markers."

    cited = extract_cited_sources(answer, local, external)

    assert len(cited.local) == 2
    assert cited.local_indices == [1, 2]
    assert len(cited.external) == 1
    assert cited.external_indices == [1]
    assert len(cited.source_files) == 2


def test_extract_empty_answer_fallback() -> None:
    local = [_local(1)]
    external = []

    cited = extract_cited_sources("", local, external)

    assert len(cited.local) == 1
    assert cited.local_indices == [1]
    assert cited.source_files[0].research_id == "r1"


def test_extract_dedupes_source_files_by_research_id() -> None:
    local = [
        Citation(
            research_id="r1",
            title="Paper",
            page=1,
            score=0.9,
            source_path="/data/researches/paper.pdf",
            filename="paper.pdf",
        ),
        Citation(
            research_id="r1",
            title="Paper",
            page=5,
            score=0.8,
            source_path="/data/researches/paper.pdf",
            filename="paper.pdf",
        ),
    ]

    cited = extract_cited_sources("Use [1] and [2].", local, [])

    assert len(cited.local) == 2
    assert len(cited.source_files) == 1
    assert cited.source_files[0].research_id == "r1"


def test_extract_preserves_first_seen_order() -> None:
    local = [_local(1), _local(2), _local(3)]
    answer = "First [3], then [1], repeat [3]."

    cited = extract_cited_sources(answer, local, [])

    assert cited.local_indices == [3, 1]
    assert [c.research_id for c in cited.local] == ["r3", "r1"]
