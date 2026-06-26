from pathlib import Path

import pytest

from research_shared.config.settings import Settings
from research_shared.ingestion.file_storage import compute_content_hash
from research_shared.ingestion.staging_storage import LocalStagingStorage


def test_staging_save_read_delete_exists(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, ingest_staging_dir=str(tmp_path / "staging"))
    storage = LocalStagingStorage(settings)
    content = b"%PDF-1.4 test"

    staged = storage.save("paper.pdf", content)

    assert staged.filename == "paper.pdf"
    assert staged.key == "paper.pdf"
    assert staged.content_hash == compute_content_hash(content)
    assert storage.exists("paper.pdf")
    assert storage.read("paper.pdf") == content

    storage.delete("paper.pdf")
    assert not storage.exists("paper.pdf")


def test_staging_cleanup_older_than(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, ingest_staging_dir=str(tmp_path))
    storage = LocalStagingStorage(settings)
    stale = tmp_path / "old.pdf"
    fresh = tmp_path / "new.pdf"
    stale.write_bytes(b"old")
    fresh.write_bytes(b"new")

    now = 1_000_000.0
    monkeypatch.setattr("research_shared.ingestion.staging_storage.time.time", lambda: now)

    import os

    os.utime(stale, (now - 100_000, now - 100_000))
    os.utime(fresh, (now - 100, now - 100))

    result = storage.cleanup_older_than(hours=24)

    assert result["deleted"] == 1
    assert "old.pdf" in result["files"]
    assert fresh.exists()
    assert not stale.exists()
