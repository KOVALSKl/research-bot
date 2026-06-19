import pytest

from research_shared.domain.models import (
    AskQuery,
    Citation,
    ResearchChunk,
    SearchResult,
    SearchType,
)
from research_shared.rag.citations import citation_display_name, dedupe_citations
from research_shared.rag.service import RagService


class _FakeSearcher:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def search(self, query):
        return self._results


class _RecordingLLM:
    def __init__(self, answer: str = "generated answer") -> None:
        self.calls: list[tuple[str, str]] = []
        self._answer = answer

    def generate(self, question: str, context: str) -> str:
        self.calls.append((question, context))
        return self._answer


def _result(
    *,
    research_id: str = "r1",
    title: str = "Paper One",
    text: str = "relevant body",
    page: int = 7,
    score: float = 0.83,
    source_path: str | None = None,
) -> SearchResult:
    return SearchResult(
        chunk=ResearchChunk(
            research_id=research_id,
            title=title,
            text=text,
            source_path=source_path,
            metadata={"page": page},
        ),
        score=score,
        search_type=SearchType.HYBRID,
    )


@pytest.mark.asyncio
async def test_rag_without_llm_returns_null_answer() -> None:
    service = RagService(_FakeSearcher([_result()]))

    response = await service.ask(AskQuery(question="what?", limit=3))

    assert response.answer is None
    assert len(response.context_chunks) == 1
    assert response.citations[0].research_id == "r1"
    assert response.citations[0].page == 7
    assert response.citations[0].score == 0.83


@pytest.mark.asyncio
async def test_rag_with_llm_returns_answer_and_passes_context() -> None:
    llm = _RecordingLLM()
    service = RagService(_FakeSearcher([_result()]), llm_provider=llm)

    response = await service.ask(AskQuery(question="what?", limit=3))

    assert response.answer == "generated answer"
    assert len(llm.calls) == 1
    question, context = llm.calls[0]
    assert question == "what?"
    assert "relevant body" in context
    assert len(response.citations) == 1


@pytest.mark.asyncio
async def test_rag_dedupes_citations_by_page_and_file() -> None:
    results = [
        _result(text="chunk a", score=0.9, source_path="/data/researches/paper.pdf"),
        _result(text="chunk b", score=0.7, source_path="/data/researches/paper.pdf"),
        _result(text="chunk c", score=0.6, page=8, source_path="/data/researches/paper.pdf"),
    ]
    service = RagService(_FakeSearcher(results))

    response = await service.ask(AskQuery(question="what?", limit=5))

    assert len(response.citations) == 2
    assert response.citations[0].filename == "paper.pdf"
    assert response.citations[0].score == 0.9
    assert response.citations[1].page == 8
    assert len(response.source_files) == 1
    assert response.source_files[0].filename == "paper.pdf"


def test_dedupe_citations_keeps_max_score() -> None:
    citations = [
        Citation(
            research_id="r1",
            title="Paper",
            page=2,
            score=0.5,
            source_path="/data/researches/paper.pdf",
        ),
        Citation(
            research_id="r1",
            title="Paper",
            page=2,
            score=0.9,
            source_path="/data/researches/paper.pdf",
        ),
    ]

    deduped = dedupe_citations(citations)

    assert len(deduped) == 1
    assert deduped[0].score == 0.9
    assert deduped[0].filename == "paper.pdf"


def test_citation_display_name_priority() -> None:
    citation = Citation(
        research_id="r1",
        title="PDF Title",
        page=1,
        score=0.5,
        source_path="/data/researches/paper_hash.pdf",
        filename="paper_hash.pdf",
        display_name="Attention Is All You Need",
    )
    assert citation_display_name(citation) == "Attention Is All You Need"


@pytest.mark.asyncio
async def test_rag_propagates_display_name() -> None:
    result = SearchResult(
        chunk=ResearchChunk(
            research_id="r1",
            title="PDF Title",
            text="body",
            source_path="/data/researches/paper.pdf",
            display_name="Custom Name",
            metadata={"page": 2},
        ),
        score=0.8,
        search_type=SearchType.HYBRID,
    )
    service = RagService(_FakeSearcher([result]))
    response = await service.ask(AskQuery(question="what?", limit=3))

    assert response.citations[0].display_name == "Custom Name"
    assert response.source_files[0].display_name == "Custom Name"


@pytest.mark.asyncio
async def test_rag_dedupes_duplicate_llm_answer() -> None:
    duplicate = "Same paragraph.\n\nSame paragraph."
    llm = _RecordingLLM(answer=duplicate)
    service = RagService(_FakeSearcher([_result()]), llm_provider=llm)

    response = await service.ask(AskQuery(question="what?", limit=3))

    assert response.answer == "Same paragraph."


@pytest.mark.asyncio
async def test_rag_context_uses_unique_numbering() -> None:
    results = [
        _result(text="chunk a", score=0.9, source_path="/data/researches/paper.pdf"),
        _result(text="chunk b", score=0.7, source_path="/data/researches/paper.pdf"),
    ]
    llm = _RecordingLLM()
    service = RagService(_FakeSearcher(results), llm_provider=llm)

    await service.ask(AskQuery(question="what?", limit=5))

    _, context = llm.calls[0]
    assert context.count("[1]") == 1
    assert "[2]" not in context
