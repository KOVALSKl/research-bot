from research_shared.literature.models import ExternalPaper
from research_shared.literature.ranking import (
    post_filter_external_papers,
    rerank_external_papers,
    rerank_external_papers_multi_query,
    score_external_paper_multi_query,
)


def test_rerank_prefers_pdf_and_title_overlap() -> None:
    papers = [
        ExternalPaper(
            title="Unrelated topic",
            url="https://example.org/a",
            source="openalex",
            abstract="Something else entirely.",
        ),
        ExternalPaper(
            title="Financial pyramid modeling",
            url="https://example.org/b",
            pdf_url="https://example.org/b.pdf",
            source="arxiv",
            abstract="Modeling financial pyramid activity.",
        ),
    ]

    ranked = rerank_external_papers(papers, "financial pyramid modeling")

    assert ranked[0].title == "Financial pyramid modeling"


def test_post_filter_keeps_pdf_without_abstract() -> None:
    papers = [
        ExternalPaper(
            title="Sparse",
            url="https://example.org/sparse",
            source="openalex",
            abstract="",
        ),
        ExternalPaper(
            title="Open PDF",
            url="https://example.org/pdf",
            pdf_url="https://example.org/pdf/file.pdf",
            source="arxiv",
            abstract="",
        ),
    ]

    filtered = post_filter_external_papers(papers, "query")

    assert len(filtered) == 1
    assert filtered[0].pdf_url is not None


def test_multi_query_score_prefers_secondary_query_match() -> None:
    paper = ExternalPaper(
        title="Graph neural networks for drug discovery",
        url="https://example.org/gnn",
        pdf_url="https://example.org/gnn.pdf",
        source="openalex",
        abstract="Applying graph neural networks to molecular property prediction.",
    )
    primary_score = score_external_paper_multi_query(paper, ["financial pyramid modeling"])
    secondary_score = score_external_paper_multi_query(
        paper,
        ["financial pyramid modeling", "graph neural networks drug discovery"],
    )
    assert secondary_score > primary_score


def test_multi_query_rerank_changes_order() -> None:
    papers = [
        ExternalPaper(
            title="Financial pyramid modeling",
            url="https://example.org/a",
            source="openalex",
            abstract="Modeling financial pyramid activity.",
        ),
        ExternalPaper(
            title="Graph neural networks for drug discovery",
            url="https://example.org/b",
            pdf_url="https://example.org/b.pdf",
            source="arxiv",
            abstract="Drug discovery with graph neural networks.",
        ),
    ]
    ranked = rerank_external_papers_multi_query(
        papers,
        ["financial pyramid modeling", "graph neural networks drug discovery"],
    )
    assert ranked[0].title == "Graph neural networks for drug discovery"
