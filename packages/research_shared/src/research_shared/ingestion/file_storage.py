from pathlib import Path

from research_shared.config.settings import Settings
from research_shared.ingestion.storage_protocol import PdfFileInfo, StoredFile


def compute_content_hash(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


def compute_research_id(content: bytes) -> str:
    """Deterministic research id derived from file content (first 16 hex chars)."""
    return compute_content_hash(content)[:16]


class LocalDocumentStorage:
    """Stores raw documents in the ``researches/`` directory (dev/local fallback)."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or Settings()
        self._dir = Path(settings.researches_dir)

    @property
    def directory(self) -> Path:
        return self._dir

    def save(self, filename: str, content: bytes) -> StoredFile:
        self._dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name
        path = self._dir / safe_name
        path.write_bytes(content)
        return StoredFile(
            path=path,
            filename=safe_name,
            content_hash=compute_content_hash(content),
            research_id=compute_research_id(content),
        )

    def list_pdfs(self) -> list[PdfFileInfo]:
        if not self._dir.exists():
            return []
        items: list[PdfFileInfo] = []
        for path in sorted(p for p in self._dir.glob("*.pdf") if p.is_file()):
            stat = path.stat()
            content = path.read_bytes()
            items.append(
                PdfFileInfo(
                    filename=path.name,
                    size=stat.st_size,
                    modified=None,
                    content_hash=compute_content_hash(content),
                )
            )
        return items

    def read(self, filename: str) -> bytes:
        path = self._dir / Path(filename).name
        if not path.is_file():
            raise FileNotFoundError(filename)
        return path.read_bytes()

    def describe(self, filename: str | Path) -> StoredFile:
        """Compute deterministic hashes/research_id for an existing file."""
        path = Path(filename)
        if not path.is_file():
            path = self._dir / path.name
        content = path.read_bytes()
        return StoredFile(
            path=path,
            filename=path.name,
            content_hash=compute_content_hash(content),
            research_id=compute_research_id(content),
        )

    def delete(self, filename: str) -> None:
        path = self._dir / Path(filename).name
        if path.is_file():
            path.unlink()

    def list(self) -> list[Path]:
        """Legacy helper — returns local PDF paths."""
        if not self._dir.exists():
            return []
        return sorted(p for p in self._dir.glob("*.pdf") if p.is_file())


# Backward-compatible alias
FileStorage = LocalDocumentStorage
