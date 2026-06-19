import hashlib
from dataclasses import dataclass
from pathlib import Path

from research_shared.config.settings import Settings


@dataclass
class StoredFile:
    path: Path
    filename: str
    content_hash: str
    research_id: str


def compute_content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def compute_research_id(content: bytes) -> str:
    """Deterministic research id derived from file content (first 16 hex chars)."""
    return compute_content_hash(content)[:16]


class FileStorage:
    """Stores raw documents in the ``researches/`` directory (source of truth)."""

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

    def list(self) -> list[Path]:
        if not self._dir.exists():
            return []
        return sorted(p for p in self._dir.glob("*.pdf") if p.is_file())

    def describe(self, path: str | Path) -> StoredFile:
        """Compute deterministic hashes/research_id for an existing file."""
        path = Path(path)
        content = path.read_bytes()
        return StoredFile(
            path=path,
            filename=path.name,
            content_hash=compute_content_hash(content),
            research_id=compute_research_id(content),
        )
