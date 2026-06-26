from unittest.mock import AsyncMock, MagicMock

import pymupdf
import pytest

from research_shared.config.settings import Settings
from research_shared.domain.models import DocumentRecord, IngestStatus, ResearchChunk
from research_shared.ingestion.chunker import RecursiveChunker
from research_shared.ingestion.file_storage import StoredFile, compute_content_hash
from research_shared.ingestion.staging_storage import StagedFile
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
    settings = Settings(_env_file=None, chunk_size=60, chunk_overlap=20, chunk_min_chars=0)
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


def test_chunker_normalizes_pdf_line_breaks_and_hyphenation() -> None:
    settings = Settings(_env_file=None, chunk_size=200, chunk_overlap=40, chunk_min_chars=20)
    chunker = RecursiveChunker(settings)
    raw = (
        "Исследование финан-\n"
        "совых пирамид показывает\n"
        "устойчивые закономерности.\n\n"
        "Вторая часть текста на новой строке."
    )
    document = ParsedDocument(title="Doc", pages=[ParsedPage(page=1, text=raw)])
    chunks = chunker.chunk(document, research_id="r1")

    assert chunks
    assert "финансовых" in chunks[0].text
    assert "финан-\n" not in chunks[0].text
    assert all(len(chunk.text) <= settings.chunk_size for chunk in chunks)


def test_chunker_skips_tiny_fragments() -> None:
    settings = Settings(_env_file=None, chunk_size=500, chunk_overlap=50, chunk_min_chars=80)
    chunker = RecursiveChunker(settings)
    document = ParsedDocument(
        title="Doc",
        pages=[ParsedPage(page=1, text="Короткий.")],
    )
    assert chunker.chunk(document, research_id="r1") == []


# --- IngestionPipeline -----------------------------------------------------


class _FakeStagingStorage:
    def __init__(self, content: bytes, filename: str = "a.pdf") -> None:
        self._content = content
        self._filename = filename
        self.deleted: list[str] = []

    def save(self, filename: str, content: bytes) -> StagedFile:
        from pathlib import Path

        return StagedFile(
            key=Path(filename).name,
            filename=Path(filename).name,
            content_hash="h1",
            research_id="r1",
            path=Path("/tmp/staged.pdf"),
        )

    def read(self, key: str) -> bytes:
        return self._content

    def delete(self, key: str) -> None:
        self.deleted.append(key)

    def exists(self, key: str) -> bool:
        return True


class _FakeArchiveStorage:
    def __init__(self) -> None:
        self.saved: list[tuple[str, bytes]] = []
        self.should_fail = False

    def save(self, filename: str, content: bytes) -> StoredFile:
        if self.should_fail:
            raise RuntimeError("archive failed")
        self.saved.append((filename, content))
        return StoredFile(
            path=None,
            filename=filename,
            content_hash="h1",
            research_id="r1",
        )

    def describe(self, filename: str) -> StoredFile:
        return StoredFile(path=None, filename=filename, content_hash="h1", research_id="r1")

    def list_pdfs(self):
        return []

    def read(self, filename: str) -> bytes:
        return b"%PDF-1.4"

    def delete(self, filename: str) -> None:
        return None


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


def _pipeline(vector_store, state_store, content=b"%PDF-1.4", n_chunks=3) -> tuple[IngestionPipeline, _FakeStagingStorage, _FakeArchiveStorage]:
    staging = _FakeStagingStorage(content)
    archive = _FakeArchiveStorage()
    pipeline = IngestionPipeline(
        parser=_FakeParser(),
        chunker=_FakeChunker(n_chunks),
        vector_store=vector_store,
        state_store=state_store,
        staging_storage=staging,
        archive_storage=archive,
        settings=Settings(_env_file=None),
    )
    return pipeline, staging, archive


@pytest.mark.asyncio
async def test_pipeline_indexes_new_document() -> None:
    vector_store = _FakeVectorStore()
    state_store = _FakeStateStore()
    pipeline, staging, archive = _pipeline(vector_store, state_store)

    result = await pipeline.process("a.pdf")

    assert result.status == IngestStatus.INDEXED
    assert result.chunk_count == 3
    assert result.skipped is False
    assert len(vector_store.upserts) == 1
    assert archive.saved == [("a.pdf", b"%PDF-1.4")]
    assert staging.deleted == ["a.pdf"]
    assert state_store.records["a.pdf"].status == IngestStatus.INDEXED


@pytest.mark.asyncio
async def test_pipeline_stores_display_name() -> None:
    vector_store = _FakeVectorStore()
    state_store = _FakeStateStore()
    pipeline, _, _ = _pipeline(vector_store, state_store)

    result = await pipeline.process("a.pdf", display_name="My Custom Title")

    assert result.status == IngestStatus.INDEXED
    assert state_store.records["a.pdf"].display_name == "My Custom Title"
    assert all(chunk.display_name == "My Custom Title" for chunk in vector_store.upserts[0])


@pytest.mark.asyncio
async def test_pipeline_skips_unchanged_indexed_document() -> None:
    content = b"%PDF-1.4"
    vector_store = _FakeVectorStore()
    state_store = _FakeStateStore()
    state_store.records["a.pdf"] = DocumentRecord(
        filename="a.pdf",
        content_hash=compute_content_hash(content),
        research_id="r1",
        status=IngestStatus.INDEXED,
        chunk_count=3,
    )
    pipeline, staging, archive = _pipeline(vector_store, state_store, content=content)

    result = await pipeline.process("a.pdf")

    assert result.skipped is True
    assert result.chunk_count == 3
    assert vector_store.upserts == []
    assert archive.saved == []
    assert staging.deleted == ["a.pdf"]


@pytest.mark.asyncio
async def test_pipeline_reindexes_changed_document() -> None:
    vector_store = _FakeVectorStore()
    state_store = _FakeStateStore()
    state_store.records["a.pdf"] = DocumentRecord(
        filename="a.pdf",
        content_hash="h1",
        research_id="r1",
        status=IngestStatus.INDEXED,
        chunk_count=3,
    )
    pipeline, _, _ = _pipeline(vector_store, state_store, content=b"%PDF-changed")

    result = await pipeline.process("a.pdf")

    assert result.status == IngestStatus.INDEXED
    assert "r1" in vector_store.deleted
    assert len(vector_store.upserts) == 1


@pytest.mark.asyncio
async def test_pipeline_marks_failed_on_error() -> None:
    state_store = _FakeStateStore()

    class _BoomVectorStore(_FakeVectorStore):
        async def upsert(self, chunks) -> int:
            raise RuntimeError("boom")

    vector_store = _BoomVectorStore()
    pipeline, staging, archive = _pipeline(vector_store, state_store)

    with pytest.raises(RuntimeError):
        await pipeline.process("a.pdf")

    assert state_store.records["a.pdf"].status == IngestStatus.FAILED
    assert state_store.records["a.pdf"].error == "boom"
    assert archive.saved == []
    assert staging.deleted == []


@pytest.mark.asyncio
async def test_pipeline_archive_failure_keeps_indexed() -> None:
    vector_store = _FakeVectorStore()
    state_store = _FakeStateStore()
    pipeline, staging, archive = _pipeline(vector_store, state_store)
    archive.should_fail = True

    result = await pipeline.process("a.pdf")

    assert result.status == IngestStatus.INDEXED
    assert result.archive_error == "archive failed"
    assert state_store.records["a.pdf"].status == IngestStatus.INDEXED
    assert state_store.records["a.pdf"].archive_error == "archive failed"
    assert staging.deleted == []


@pytest.mark.asyncio
async def test_pipeline_archive_runs_after_upsert() -> None:
    vector_store = _FakeVectorStore()
    state_store = _FakeStateStore()
    pipeline, _, archive = _pipeline(vector_store, state_store)

    class _TrackingArchive(_FakeArchiveStorage):
        def save(self, filename: str, content: bytes) -> StoredFile:
            assert vector_store.upserts, "archive must run after upsert"
            return super().save(filename, content)

    pipeline._archive_storage = _TrackingArchive()
    await pipeline.process("a.pdf")


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
