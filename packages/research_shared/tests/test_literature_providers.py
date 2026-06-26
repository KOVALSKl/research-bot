import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from research_shared.literature.arxiv import ArxivLiteratureProvider
from research_shared.literature.openalex import OpenAlexLiteratureProvider
from research_shared.literature.semantic_scholar import SemanticScholarLiteratureProvider
from research_shared.config.settings import Settings

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "literature"


def _mock_response(content: str | dict, *, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    if isinstance(content, dict):
        response.json.return_value = content
        response.text = json.dumps(content)
    else:
        response.text = content
        response.json.side_effect = ValueError("not json")
    return response


@pytest.mark.asyncio
async def test_openalex_maps_works_from_fixture() -> None:
    payload = json.loads((FIXTURES / "openalex_works_response.json").read_text())
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _mock_response(payload)

    provider = OpenAlexLiteratureProvider(client=client)
    papers = await provider.search("gnn bankruptcy", limit=10)

    assert len(papers) == 2
    assert papers[0].title == "Graph Neural Networks for Financial Distress"
    assert papers[0].authors == ["Alice Smith", "Bob Jones"]
    assert papers[0].year == 2023
    assert papers[0].doi == "10.1234/gnn.fin.2023"
    assert papers[0].source == "openalex"
    assert papers[0].pdf_url == "https://example.org/papers/gnn-financial.pdf"
    assert "bankruptcy" in papers[0].abstract


@pytest.mark.asyncio
async def test_openalex_empty_results() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _mock_response({"results": []})

    provider = OpenAlexLiteratureProvider(client=client)
    papers = await provider.search("nothing", limit=5)

    assert papers == []


@pytest.mark.asyncio
async def test_openalex_malformed_work_skipped() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _mock_response({"results": [{"display_name": ""}, "bad"]})

    provider = OpenAlexLiteratureProvider(client=client)
    papers = await provider.search("q", limit=5)

    assert papers == []


@pytest.mark.asyncio
async def test_openalex_year_from_filter_param() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _mock_response({"results": []})

    provider = OpenAlexLiteratureProvider(client=client)
    await provider.search("q", limit=3, year_from=2020)

    params = client.get.call_args.kwargs["params"]
    assert params["filter"] == "from_publication_date:2020-01-01"


@pytest.mark.asyncio
async def test_arxiv_parses_atom_fixture() -> None:
    xml_text = (FIXTURES / "arxiv_atom_feed.xml").read_text()
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _mock_response(xml_text)

    provider = ArxivLiteratureProvider(client=client)
    papers = await provider.search("credit risk", limit=10)

    assert len(papers) == 2
    assert papers[0].title == "Deep Learning for SME Credit Risk"
    assert papers[0].year == 2023
    assert papers[0].doi == "10.1234/arxiv.sme.2023"
    assert papers[0].source == "arxiv"
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2301.00001.pdf"
    assert papers[1].abstract == ""


@pytest.mark.asyncio
async def test_arxiv_empty_feed() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _mock_response(
        '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    )

    provider = ArxivLiteratureProvider(client=client)
    papers = await provider.search("q", limit=5)

    assert papers == []


@pytest.mark.asyncio
async def test_arxiv_year_from_filters_results() -> None:
    xml_text = (FIXTURES / "arxiv_atom_feed.xml").read_text()
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _mock_response(xml_text)

    provider = ArxivLiteratureProvider(client=client)
    papers = await provider.search("credit", limit=10, year_from=2022)

    assert len(papers) == 1
    assert papers[0].year == 2023


@pytest.mark.asyncio
async def test_semantic_scholar_without_api_key_returns_empty() -> None:
    settings = Settings(_env_file=None, semantic_scholar_api_key="")
    provider = SemanticScholarLiteratureProvider(settings)
    client = AsyncMock(spec=httpx.AsyncClient)

    papers = await provider.search("gnn", limit=5)

    assert papers == []
    client.get.assert_not_called()


@pytest.mark.asyncio
async def test_semantic_scholar_maps_fixture_with_key() -> None:
    payload = json.loads((FIXTURES / "semantic_scholar_papers.json").read_text())
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = _mock_response(payload)
    settings = Settings(_env_file=None, semantic_scholar_api_key="test-key")

    provider = SemanticScholarLiteratureProvider(settings, client=client)
    papers = await provider.search("gnn", limit=5)

    assert len(papers) == 2
    assert papers[0].doi == "10.1234/gnn.fin.2023"
    assert papers[0].source == "semantic_scholar"
    headers = client.get.call_args.kwargs["headers"]
    assert headers["x-api-key"] == "test-key"


@pytest.mark.asyncio
async def test_semantic_scholar_http_error_returns_empty() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.side_effect = httpx.HTTPError("rate limit")
    settings = Settings(_env_file=None, semantic_scholar_api_key="test-key")

    provider = SemanticScholarLiteratureProvider(settings, client=client)
    papers = await provider.search("gnn", limit=5)

    assert papers == []
