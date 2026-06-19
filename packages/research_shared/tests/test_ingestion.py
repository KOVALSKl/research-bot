from unittest.mock import AsyncMock, MagicMock

import pymupdf
import pytest

from research_shared.config.settings import Settings
from research_shared.domain.models import DocumentRecord, IngestStatus, ResearchChunk
from research_shared.ingestion.chunker import RecursiveChunker
from research_shared.ingestion.file_storage import StoredFile
from research_shared.ingestion.pdf_parser import PyMuPDFParser
from research_shared.ingestion.pipeline import IngestionPipeline
from research_shared.ingestion.protocols import ParsedDocument, ParsedPage
from research_shared.ingestion.state_store import QdrantIngestionStateStore


# --- PyMuPDFParser ---------------------------------------------------------


def _make_pdf(path, pages: list[str]) -> None:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_pdf_parser_pages_and_title(tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, ["First page body", "Second page body"])

    document = PyMuPDFParser().parse(pdf_path)

    assert len(document.pages) == 2
    assert [p.page for p in document.pages] == [1, 2]
    assert "First page" in document.pages[0].text
    assert "Second page" in document.pages[1].text
    # No PDF title metadata → falls back to filename stem.
    assert document.title == "paper"


# --- RecursiveChunker ------------------------------------------------------


def test_chunker_page_and_chunk_index_and_overlap() -> None:
    settings = Settings(_env_file=None, chunk_size=60, chunk_overlap=20)
    chunker = RecursiveChunker(settings)

    long_text = " ".join(f"word{i}" for i in range(40))
    document = ParsedDocument(
        title="Doc",
        pages=[ParsedPage(page=1, text=long_text), ParsedPage(page=2, text="short tail")],
    )

    chunks = chunker.chunk(document, research_id="r1")

    assert len(chunks) >= 2
    assert all(isinstance(c, ResearchChunk) for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all("page" in c.metadata for c in chunks)
    assert all(len(c.text) <= settings.chunk_size for c in chunks)
    # Page binding: the last chunk comes from page 2.
    assert chunks[-1].metadata["page"] == 2
    assert all(c.research_id == "r1" for c in chunks)

    # Overlap: consecutive chunks on the same page share at least one token.
    page1_chunks = [c for c in chunks if c.metadata["page"] == 1]
    if len(page1_chunks) >= 2:
        first_tokens = set(page1_chunks[0].text.split())
        second_tokens = set(page1_chunks[1].text.split())
        assert first_tokens & second_tokens


# --- IngestionPipeline -----------------------------------------------------


class _FakeFileStorage:
    def __init__(self, stored: StoredFile) -> None:
        self._stored = stored

    def describe(self, path) -> StoredFile:
        return self._stored


class _FakeParser:
    def parse(self, path) -> ParsedDocument:
        return ParsedDocument(title="T", pages=[ParsedPage(page=1, text="body")])


class _FakeChunker:
    def __init__(self, n: int = 3) -> None:
        self._n = n

    def chunk(self, document, research_id) -> list[ResearchChunk]:
        return [
            ResearchChunk(research_id=research_id, title="T", text=f"c{i}", metadata={"page": 1})
            for i in range(self._n)
        ]


class _FakeVectorStore:
    def __init__(self) -> None:
        self.upserts: list[list[ResearchChunk]] = []
        self.deleted: list[str] = []

    async def upsert(self, chunks) -> int:
        self.upserts.append(chunks)
        return len(chunks)

    async def delete_by_ids(self, ids) -> int:
        return len(ids)

    async def delete_by_research_id(self, research_id) -> int:
        self.deleted.append(research_id)
        return 1


class _FakeStateStore:
    def __init__(self) -> None:
        self.records: dict[str, DocumentRecord] = {}

    async def ensure_collection(self) -> None:
        return None

    async def get(self, filename) -> DocumentRecord | None:
        return self.records.get(filename)

    async def upsert(self, record: DocumentRecord) -> None:
        self.records[record.filename] = record

    async def list(self) -> list[DocumentRecord]:
        return list(self.records.values())


def _pipeline(vector_store, state_store, stored, n_chunks=3) -> IngestionPipeline:
    return IngestionPipeline(
        parser=_FakeParser(),
        chunker=_FakeChunker(n_chunks),
        vector_store=vector_store,
        state_store=state_store,
        file_storage=_FakeFileStorage(stored),
        settings=Settings(_env_file=None),
    )


@pytest.mark.asyncio
async def test_pipeline_indexes_new_document() -> None:
    stored = StoredFile(path="researches/a.pdf", filename="a.pdf", content_hash="h1", research_id="r1")
    vector_store = _FakeVectorStore()
    state_store = _FakeStateStore()
    pipeline = _pipeline(vector_store, state_store, stored)

    result = await pipeline.process("researches/a.pdf")

    assert result.status == IngestStatus.INDEXED
    assert result.chunk_count == 3
    assert result.skipped is False
    assert len(vector_store.upserts) == 1
    assert state_store.records["a.pdf"].status == IngestStatus.INDEXED


@pytest.mark.asyncio
async def test_pipeline_stores_display_name() -> None:
    stored = StoredFile(path="researches/a.pdf", filename="a.pdf", content_hash="h1", research_id="r1")
    vector_store = _FakeVectorStore()
    state_store = _FakeStateStore()
    pipeline = _pipeline(vector_store, state_store, stored)

    result = await pipeline.process("researches/a.pdf", display_name="My Custom Title")

    assert result.status == IngestStatus.INDEXED
    assert state_store.records["a.pdf"].display_name == "My Custom Title"
    assert all(chunk.display_name == "My Custom Title" for chunk in vector_store.upserts[0])


@pytest.mark.asyncio
async def test_pipeline_skips_unchanged_indexed_document() -> None:
    stored = StoredFile(path="researches/a.pdf", filename="a.pdf", content_hash="h1", research_id="r1")
    vector_store = _FakeVectorStore()
    state_store = _FakeStateStore()
    state_store.records["a.pdf"] = DocumentRecord(
        filename="a.pdf",
        content_hash="h1",
        research_id="r1",
        status=IngestStatus.INDEXED,
        chunk_count=3,
    )
    pipeline = _pipeline(vector_store, state_store, stored)

    result = await pipeline.process("researches/a.pdf")

    assert result.skipped is True
    assert result.chunk_count == 3
    assert vector_store.upserts == []  # no re-index


@pytest.mark.asyncio
async def test_pipeline_reindexes_changed_document() -> None:
    stored = StoredFile(path="researches/a.pdf", filename="a.pdf", content_hash="h2", research_id="r2")
    vector_store = _FakeVectorStore()
    state_store = _FakeStateStore()
    state_store.records["a.pdf"] = DocumentRecord(
        filename="a.pdf",
        content_hash="h1",
        research_id="r1",
        status=IngestStatus.INDEXED,
        chunk_count=3,
    )
    pipeline = _pipeline(vector_store, state_store, stored)

    result = await pipeline.process("researches/a.pdf")

    assert result.status == IngestStatus.INDEXED
    assert result.research_id == "r2"
    # Old chunks (r1) and current (r2) both cleared before re-index.
    assert "r1" in vector_store.deleted
    assert "r2" in vector_store.deleted
    assert len(vector_store.upserts) == 1


@pytest.mark.asyncio
async def test_pipeline_marks_failed_on_error() -> None:
    stored = StoredFile(path="researches/a.pdf", filename="a.pdf", content_hash="h1", research_id="r1")
    state_store = _FakeStateStore()

    class _BoomVectorStore(_FakeVectorStore):
        async def upsert(self, chunks) -> int:
            raise RuntimeError("boom")

    vector_store = _BoomVectorStore()
    pipeline = _pipeline(vector_store, state_store, stored)

    with pytest.raises(RuntimeError):
        await pipeline.process("researches/a.pdf")

    assert state_store.records["a.pdf"].status == IngestStatus.FAILED
    assert state_store.records["a.pdf"].error == "boom"


# --- QdrantIngestionStateStore --------------------------------------------


@pytest.mark.asyncio
async def test_state_store_ensure_creates_collection_when_missing() -> None:
    client = AsyncMock()
    client.collection_exists = AsyncMock(return_value=False)
    client.create_collection = AsyncMock()

    store = QdrantIngestionStateStore(client, Settings(_env_file=None))
    await store.ensure_collection()

    client.create_collection.assert_awaited_once()


@pytest.mark.asyncio
async def test_state_store_get_returns_record() -> None:
    point = MagicMock()
    point.payload = {
        "filename": "a.pdf",
        "content_hash": "h1",
        "research_id": "r1",
        "status": "indexed",
        "chunk_count": 5,
    }
    client = AsyncMock()
    client.retrieve = AsyncMock(return_value=[point])

    store = QdrantIngestionStateStore(client, Settings(_env_file=None))
    record = await store.get("a.pdf")

    assert record is not None
    assert record.filename == "a.pdf"
    assert record.status == IngestStatus.INDEXED
    assert record.chunk_count == 5


@pytest.mark.asyncio
async def test_state_store_upsert_writes_point() -> None:
    client = AsyncMock()
    client.upsert = AsyncMock()

    store = QdrantIngestionStateStore(client, Settings(_env_file=None))
    await store.upsert(
        DocumentRecord(filename="a.pdf", content_hash="h1", research_id="r1")
    )

    client.upsert.assert_awaited_once()
