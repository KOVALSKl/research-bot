from __future__ import annotations

import re

from research_shared.literature.models import ExternalPaper
from research_shared.logging_config import get_logger

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[\w\u0400-\u04FF]+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_PATTERN.findall(text) if len(token) > 2}


def score_external_paper(paper: ExternalPaper, query: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    title_tokens = _tokenize(paper.title)
    abstract_tokens = _tokenize(paper.abstract)
    title_overlap = len(query_tokens & title_tokens) / len(query_tokens)
    abstract_overlap = len(query_tokens & abstract_tokens) / len(query_tokens)
    score = title_overlap * 2.0 + abstract_overlap

    if paper.pdf_url:
        score += 0.75
    if paper.abstract.strip():
        score += 0.25
    if paper.year is not None:
        score += min(paper.year - 1990, 30) / 300.0
    return score


def score_external_paper_multi_query(paper: ExternalPaper, queries: list[str]) -> float:
    if not queries:
        return 0.0
    return max(score_external_paper(paper, query) for query in queries if query.strip())


def rerank_external_papers(
    papers: list[ExternalPaper],
    query: str,
) -> list[ExternalPaper]:
    if not papers:
        return []
    return sorted(
        papers,
        key=lambda paper: score_external_paper(paper, query),
        reverse=True,
    )


def rerank_external_papers_multi_query(
    papers: list[ExternalPaper],
    queries: list[str],
) -> list[ExternalPaper]:
    if not papers:
        return []
    if len(queries) <= 1:
        return rerank_external_papers(papers, queries[0] if queries else "")
    return sorted(
        papers,
        key=lambda paper: score_external_paper_multi_query(paper, queries),
        reverse=True,
    )


def post_filter_external_papers(
    papers: list[ExternalPaper],
    query: str,
    *,
    min_score: float = 0.15,
) -> list[ExternalPaper]:
    filtered: list[ExternalPaper] = []
    for paper in papers:
        score = score_external_paper(paper, query)
        if paper.abstract.strip() or score >= min_score or paper.pdf_url:
            filtered.append(paper)
            continue
        logger.info(
            "External literature post-filter dropped paper",
            extra={
                "event": "literature.post_filter_dropped",
                "title": paper.title,
                "score": score,
                "has_pdf_url": bool(paper.pdf_url),
            },
        )
    return filtered
