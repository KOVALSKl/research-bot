import pytest

from vk_bot.config import VkBotSettings
from vk_bot.state.upload_naming import MemoryUploadNamingStore, NamingFile


@pytest.fixture
def store(tmp_path) -> MemoryUploadNamingStore:
  settings = VkBotSettings(vk_naming_temp_dir=str(tmp_path / "uploads"))
  return MemoryUploadNamingStore(settings)


@pytest.mark.asyncio
async def test_naming_session_save_get_clear(store: MemoryUploadNamingStore):
  files = [
    NamingFile(original_name="a.pdf", temp_path="/tmp/a.pdf", size=10),
    NamingFile(original_name="b.pdf", temp_path="/tmp/b.pdf", size=20),
  ]
  await store.save(1, peer_id=100, files=files)

  assert await store.active(1)
  session = await store.get(1)
  assert session is not None
  assert session.peer_id == 100
  assert len(session.files) == 2

  await store.clear(1)
  assert not await store.active(1)


@pytest.mark.asyncio
async def test_naming_session_append_and_advance(store: MemoryUploadNamingStore):
  files = [NamingFile(original_name="a.pdf", temp_path="/tmp/a.pdf")]
  await store.save(2, peer_id=200, files=files)

  await store.append_name(2, "Custom Name")
  session = await store.advance(2)
  assert session is not None
  assert session.names == ["Custom Name"]
  assert session.current_index == 1


@pytest.mark.asyncio
async def test_save_temp_file(store: MemoryUploadNamingStore):
  naming_file = store.save_temp_file(3, b"%PDF content", "paper.pdf")
  assert naming_file.original_name == "paper.pdf"
  assert naming_file.size == len(b"%PDF content")
