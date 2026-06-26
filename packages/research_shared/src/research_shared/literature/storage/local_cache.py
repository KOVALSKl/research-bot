from __future__ import annotations

from pathlib import Path


class LocalPdfCache:
    def __init__(self, root_dir: str) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def path_for(self, cache_key: str) -> Path:
        safe_key = cache_key.replace("/", "_")
        return self._root / f"{safe_key}.pdf"

    def exists(self, cache_key: str) -> bool:
        return self.path_for(cache_key).is_file()

    def read(self, cache_key: str) -> bytes | None:
        path = self.path_for(cache_key)
        if not path.is_file():
            return None
        return path.read_bytes()

    def write(self, cache_key: str, content: bytes) -> Path:
        path = self.path_for(cache_key)
        path.write_bytes(content)
        return path
