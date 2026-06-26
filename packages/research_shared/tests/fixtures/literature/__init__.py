"""Shared fixtures for external literature provider tests.

Fixtures live in ``tests/fixtures/literature/``:
- ``openalex_works_response.json`` — OpenAlex Works API (2 works, one with inverted abstract)
- ``arxiv_atom_feed.xml`` — arXiv Atom feed (2 entries, one without summary/DOI)
- ``semantic_scholar_papers.json`` — Semantic Scholar search (duplicate DOI with OpenAlex)
"""

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "literature"


def load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")
