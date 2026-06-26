from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from research_shared.config.settings import Settings
from research_shared.ingestion.file_storage import compute_content_hash, compute_research_id


@dataclass
class StagedFile:
    key: str
    filename: str
    content_hash: str
    research_id: str
    path: Path


@runtime_checkable
class StagingStorage(Protocol):
    def save(self, filename: str, content: bytes) -> StagedFile: ...

    def read(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...

    def exists(self, key: str) -> bool: ...


class LocalStagingStorage:
    """Transient local storage for uploads before background ingest."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or Settings()
        self._dir = Path(settings.ingest_staging_dir)

    @property
    def directory(self) -> Path:
        return self._dir

    def _path_for(self, key: str) -> Path:
        safe_key = Path(key).name
        return self._dir / safe_key

    def save(self, filename: str, content: bytes) -> StagedFile:
        self._dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name
        path = self._dir / safe_name
        path.write_bytes(content)
        return StagedFile(
            key=safe_name,
            filename=safe_name,
            content_hash=compute_content_hash(content),
            research_id=compute_research_id(content),
            path=path,
        )

    def read(self, key: str) -> bytes:
        path = self._path_for(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.read_bytes()

    def delete(self, key: str) -> None:
        path = self._path_for(key)
        if path.is_file():
            path.unlink()

    def exists(self, key: str) -> bool:
        return self._path_for(key).is_file()

    def cleanup_older_than(self, hours: float) -> dict[str, object]:
        if not self._dir.exists():
            return {"deleted": 0, "files": []}
        cutoff = time.time() - hours * 3600
        deleted: list[str] = []
        for path in self._dir.iterdir():
            if not path.is_file():
                continue
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                deleted.append(path.name)
        return {"deleted": len(deleted), "files": deleted}
