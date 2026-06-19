from pathlib import Path

import pytest
from research_shared.domain.models import AskResponse, Citation, ResearchChunk, SourceFileRef

from vk_bot.config import VkBotSettings
from vk_bot.core_api.client import CoreApiError
from vk_bot.domain import Attachment, IncomingMessage
from vk_bot.handlers.bot_service import BotService, format_ask_response
from vk_bot.handlers.router import CommandRouter
from vk_bot.security.rate_limiter import MemoryRateLimiter
from vk_bot.security.sanitizer import MessageSanitizer
from vk_bot.state.message_dedup import MemoryMessageDedupStore
from vk_bot.state.upload_naming import MemoryUploadNamingStore
from vk_bot.state.user_session import MemoryUserSessionStore


class MockVkApi:
  def __init__(self) -> None:
    self.sent: list[tuple[int, str, str | None]] = []
    self.downloaded: list[tuple[bytes, str]] = []
    self.download_raises: Exception | None = None
    self.upload_doc_result = "doc1_2"
    self.upload_doc_raises: Exception | None = None

  async def send_message(
    self,
    peer_id: int,
    text: str,
    *,
    attachment: str | None = None,
  ) -> None:
    self.sent.append((peer_id, text, attachment))

  async def download_attachments(self, attachments: list[Attachment]) -> list[tuple[bytes, str]]:
    if self.download_raises:
      raise self.download_raises
    return self.downloaded

  async def upload_doc_to_vk(
    self,
    content: bytes,
    filename: str,
    peer_id: int,
  ) -> str:
    if self.upload_doc_raises:
      raise self.upload_doc_raises
    return self.upload_doc_result


class MockCoreApi:
  def __init__(self) -> None:
    self.ask_calls: list[str] = []
    self.upload_calls: list[tuple[bytes, str, str | None]] = []
    self.batch_calls: list[tuple[list[tuple[bytes, str]], list[str] | None]] = []
    self.download_calls: list[str] = []
    self.list_response: list = []
    self.ask_response = AskResponse(answer="Test answer", citations=[])
    self.upload_response = {"task_id": "task-1", "research_id": "res-1"}
    self.batch_response = {
      "jobs": [
        {"task_id": "task-1", "filename": "a.pdf"},
        {"task_id": "task-2", "filename": "b.pdf"},
      ]
    }
    self.download_response = (b"%PDF", "paper.pdf")
    self.raise_on_ask: Exception | None = None
    self.raise_on_download: Exception | None = None

  async def ask(self, question: str, limit: int | None = None) -> AskResponse:
    if self.raise_on_ask:
      raise self.raise_on_ask
    self.ask_calls.append(question)
    return self.ask_response

  async def upload_document(
    self,
    content: bytes,
    filename: str,
    *,
    display_name: str | None = None,
  ) -> dict:
    self.upload_calls.append((content, filename, display_name))
    return self.upload_response

  async def upload_batch(
    self,
    files: list[tuple[bytes, str]],
    *,
    display_names: list[str] | None = None,
  ) -> dict:
    self.batch_calls.append((files, display_names))
    return self.batch_response

  async def list_documents(self, status: str | None = None) -> list:
    return self.list_response

  async def get_task_status(self, task_id: str) -> dict:
    return {"status": "indexed", "task_id": task_id}

  async def download_source_file(self, research_id: str) -> tuple[bytes, str]:
    if self.raise_on_download:
      raise self.raise_on_download
    self.download_calls.append(research_id)
    return self.download_response


class MockQueue:
  def __init__(self, busy: bool = False) -> None:
    self.busy = busy
    self.saved: list[dict] = []

  async def is_busy(self, user_id: int) -> bool:
    return self.busy

  async def save_batch(self, user_id: int, **kwargs) -> None:
    self.saved.append({"user_id": user_id, **kwargs})


@pytest.fixture
def settings() -> VkBotSettings:
  return VkBotSettings(
    vk_min_question_length=12,
    vk_rate_limit_backend="memory",
    vk_rate_limit_max_messages=5,
    vk_rate_limit_window_seconds=60,
    vk_debounce_seconds=0,
    vk_ask_attach_enabled=True,
  )


@pytest.fixture
def bot_service(settings: VkBotSettings) -> tuple[BotService, MockVkApi, MockCoreApi, MockQueue]:
  vk = MockVkApi()
  core = MockCoreApi()
  queue = MockQueue()
  service = BotService(
    settings=settings,
    vk_api=vk,
    core_api=core,
    rate_limiter=MemoryRateLimiter(settings),
    sanitizer=MessageSanitizer(settings),
    queue_store=queue,
    router=CommandRouter(settings),
    session_store=MemoryUserSessionStore(),
    naming_store=MemoryUploadNamingStore(settings),
    dedup_store=MemoryMessageDedupStore(),
  )
  return service, vk, core, queue


@pytest.mark.asyncio
async def test_ask_command_flow(bot_service):
  service, vk, core, _ = bot_service
  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="/ask What is quantum computing?"),
  )

  assert core.ask_calls == ["What is quantum computing?"]
  assert any("обработка" in text.lower() for _, text, _ in vk.sent)
  assert any("Test answer" in text for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_ask_attaches_source_pdf(bot_service):
  service, vk, core, _ = bot_service
  core.ask_response = AskResponse(
    answer="Answer",
    citations=[
      Citation(
        research_id="r1",
        title="Paper",
        page=2,
        score=0.9,
        source_path="/data/researches/paper.pdf",
        filename="paper.pdf",
      )
    ],
    source_files=[SourceFileRef(research_id="r1", filename="paper.pdf")],
  )

  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="/ask What is quantum computing?"),
  )

  assert core.download_calls == ["r1"]
  assert any(attachment == "doc1_2" for _, _, attachment in vk.sent)


@pytest.mark.asyncio
async def test_ask_attachment_failure_still_sends_text(bot_service):
  service, vk, core, _ = bot_service
  core.ask_response = AskResponse(
    answer="Answer",
    source_files=[SourceFileRef(research_id="r1", filename="paper.pdf")],
  )
  vk.upload_doc_raises = RuntimeError("upload failed")

  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="/ask What is quantum computing?"),
  )

  assert any("Answer" in text for _, text, _ in vk.sent)
  assert all(
    attachment is None
    for _, text, attachment in vk.sent
    if "Answer" in text
  )


@pytest.mark.asyncio
async def test_plain_text_does_not_call_ask(bot_service):
  service, vk, core, _ = bot_service
  await service._session.mark_seen(1)
  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="What is quantum computing?"),
  )

  assert core.ask_calls == []
  assert any("не понял" in text.lower() for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_short_ask_skips_core_api(bot_service):
  service, vk, core, _ = bot_service
  await service.handle(IncomingMessage(user_id=1, peer_id=1, text="/ask short"))

  assert core.ask_calls == []
  assert any("короткий" in text.lower() for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_rate_limit_blocks_with_cooldown(bot_service, settings):
  service, vk, core, _ = bot_service
  settings.vk_rate_limit_max_messages = 1
  service._rate_limiter = MemoryRateLimiter(settings)

  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="/ask First question long enough"),
  )
  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="/ask Second question long enough"),
  )

  assert len(core.ask_calls) == 1
  rate_messages = [text for _, text, _ in vk.sent if "много сообщений" in text.lower()]
  assert len(rate_messages) == 1


@pytest.mark.asyncio
async def test_outgoing_message_is_ignored(bot_service):
  service, vk, core, _ = bot_service
  await service.handle(
    IncomingMessage(
      user_id=1,
      peer_id=1,
      text="Слишком много сообщений",
      is_outgoing=True,
    ),
  )

  assert core.ask_calls == []
  assert vk.sent == []


@pytest.mark.asyncio
async def test_first_visit_sends_welcome(bot_service):
  service, vk, core, _ = bot_service
  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="What is quantum computing?"),
  )

  assert core.ask_calls == []
  assert any("бот для поиска" in text.lower() for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_greeting_sends_welcome(bot_service):
  service, vk, core, _ = bot_service
  await service.handle(IncomingMessage(user_id=2, peer_id=2, text="привет"))

  assert core.ask_calls == []
  assert sum("бот для поиска" in text.lower() for _, text, _ in vk.sent) >= 1


@pytest.mark.asyncio
async def test_first_visit_help_sends_welcome_and_help(bot_service):
  service, vk, core, _ = bot_service
  await service.handle(IncomingMessage(user_id=7, peer_id=7, text="помощь"))

  assert core.ask_calls == []
  assert any("бот для поиска" in text.lower() for _, text, _ in vk.sent)
  assert any("команды бота" in text.lower() for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_duplicate_message_is_ignored(bot_service):
  service, vk, core, _ = bot_service
  message = IncomingMessage(user_id=8, peer_id=8, message_id=999, text="привет")

  await service.handle(message)
  await service.handle(message)

  assert core.ask_calls == []
  assert sum("бот для поиска" in text.lower() for _, text, _ in vk.sent) == 1


@pytest.mark.asyncio
async def test_too_many_pdf_attachments_rejected(bot_service, settings):
  service, vk, core, _ = bot_service
  settings.vk_max_pdf_attachments = 3
  attachments = [
    Attachment(filename=f"file{i}.pdf", url=f"http://x/{i}", ext="pdf")
    for i in range(4)
  ]

  await service.handle(
    IncomingMessage(user_id=9, peer_id=9, attachments=attachments),
  )

  assert core.upload_calls == []
  assert core.batch_calls == []
  assert any("не более 3" in text.lower() for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_pdf_upload_starts_naming_wizard(bot_service):
  service, vk, core, queue = bot_service
  vk.downloaded = [(b"%PDF", "paper.pdf")]
  attachment = Attachment(filename="paper.pdf", url="http://x", ext="pdf")

  await service.handle(
    IncomingMessage(user_id=3, peer_id=3, attachments=[attachment]),
  )

  assert core.upload_calls == []
  assert any("название" in text.lower() for _, text, _ in vk.sent)

  await service.handle(
    IncomingMessage(user_id=3, peer_id=3, text="My Paper"),
  )

  assert core.upload_calls == [(b"%PDF", "paper.pdf", "My Paper")]
  assert queue.saved
  assert any("очередь" in text.lower() for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_naming_wizard_two_files_finalize_with_display_names(bot_service):
  service, vk, core, queue = bot_service
  core.upload_response = {"task_id": "task-1", "research_id": "res-1"}
  vk.downloaded = [(b"%PDF-1", "first.pdf"), (b"%PDF-2", "second.pdf")]
  attachments = [
    Attachment(filename="first.pdf", url="http://x/1", ext="pdf"),
    Attachment(filename="second.pdf", url="http://x/2", ext="pdf"),
  ]

  await service.handle(
    IncomingMessage(user_id=11, peer_id=11, message_id=1101, attachments=attachments),
  )
  await service.handle(
    IncomingMessage(user_id=11, peer_id=11, message_id=1102, text="Paper One"),
  )
  await service.handle(
    IncomingMessage(user_id=11, peer_id=11, message_id=1103, text="Paper Two"),
  )

  assert core.upload_calls == [
    (b"%PDF-1", "first.pdf", "Paper One"),
    (b"%PDF-2", "second.pdf", "Paper Two"),
  ]
  assert len(queue.saved) == 1
  assert queue.saved[0]["filenames"] == ["Paper One", "Paper Two"]
  assert not await service._naming.active(11)


@pytest.mark.asyncio
async def test_naming_cancel_clears_session(bot_service):
  service, vk, core, _ = bot_service
  vk.downloaded = [(b"%PDF", "paper.pdf")]
  attachment = Attachment(filename="paper.pdf", url="http://x", ext="pdf")

  await service.handle(
    IncomingMessage(user_id=12, peer_id=12, message_id=1201, attachments=[attachment]),
  )
  session = await service._naming.get(12)
  assert session is not None
  temp_paths = [file.temp_path for file in session.files]

  await service.handle(
    IncomingMessage(user_id=12, peer_id=12, message_id=1202, text="/cancel"),
  )

  assert not await service._naming.active(12)
  assert core.upload_calls == []
  assert any("отмен" in text.lower() for _, text, _ in vk.sent)
  for temp_path in temp_paths:
    assert not Path(temp_path).is_file()


@pytest.mark.asyncio
async def test_naming_rejects_pdf_while_active(bot_service):
  service, vk, core, _ = bot_service
  vk.downloaded = [(b"%PDF", "paper.pdf")]
  attachment = Attachment(filename="paper.pdf", url="http://x", ext="pdf")

  await service.handle(
    IncomingMessage(user_id=13, peer_id=13, message_id=1301, attachments=[attachment]),
  )
  vk.sent.clear()
  vk.downloaded = [(b"%PDF-2", "other.pdf")]

  await service.handle(
    IncomingMessage(
      user_id=13,
      peer_id=13,
      message_id=1302,
      attachments=[Attachment(filename="other.pdf", url="http://x/2", ext="pdf")],
    ),
  )

  assert core.upload_calls == []
  assert any("завершите ввод названий" in text.lower() for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_handle_list_command(bot_service):
  from research_shared.domain.models import DocumentListItem, IngestStatus

  service, vk, core, _ = bot_service
  await service._session.mark_seen(14)
  core.list_response = [
    DocumentListItem(
      research_id="r1",
      filename="paper_abc.pdf",
      display_name="Attention Is All You Need",
      status=IngestStatus.INDEXED,
      chunk_count=42,
    )
  ]

  await service.handle(
    IncomingMessage(user_id=14, peer_id=14, text="/list"),
  )

  assert any("Attention Is All You Need" in text for _, text, _ in vk.sent)
  assert any("42 чанков" in text for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_pdf_upload_dash_uses_original_name(bot_service):
  service, vk, core, _ = bot_service
  vk.downloaded = [(b"%PDF", "paper.pdf")]
  attachment = Attachment(filename="paper.pdf", url="http://x", ext="pdf")

  await service.handle(
    IncomingMessage(user_id=10, peer_id=10, attachments=[attachment]),
  )
  await service.handle(
    IncomingMessage(user_id=10, peer_id=10, text="-"),
  )

  assert core.upload_calls == [(b"%PDF", "paper.pdf", "paper.pdf")]


@pytest.mark.asyncio
async def test_empty_pdf_download(bot_service):
  service, vk, core, _ = bot_service
  attachment = Attachment(filename="paper.pdf", url="http://x", ext="pdf")

  await service.handle(
    IncomingMessage(user_id=4, peer_id=4, attachments=[attachment]),
  )

  assert core.upload_calls == []
  assert any("не удалось скачать" in text.lower() for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_non_pdf_attachment_without_text(bot_service):
  service, vk, core, _ = bot_service
  attachment = Attachment(filename="photo.jpg", url="http://x", ext="jpg")

  await service.handle(
    IncomingMessage(user_id=5, peer_id=5, attachments=[attachment]),
  )

  assert core.upload_calls == []
  assert any("только pdf" in text.lower() for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_busy_queue_rejects_upload(bot_service):
  service, vk, core, queue = bot_service
  queue.busy = True
  vk.downloaded = [(b"%PDF", "paper.pdf")]
  attachment = Attachment(filename="paper.pdf", url="http://x", ext="pdf")

  await service.handle(
    IncomingMessage(user_id=6, peer_id=6, attachments=[attachment]),
  )

  assert core.upload_calls == []
  assert any("предыдущая" in text.lower() for _, text, _ in vk.sent)


def test_format_ask_response_with_answer():
  response = AskResponse(
    answer="Hello",
    citations=[
      Citation(
        research_id="r1",
        title="Attention",
        page=29,
        score=0.9,
        source_path="/data/researches/paper.pdf",
        filename="paper.pdf",
        authors=["Vaswani"],
      )
    ],
  )
  text = format_ask_response(response)
  assert "Hello" in text
  assert "paper.pdf, стр. 29 (Vaswani)" in text


def test_format_ask_response_with_display_name():
  response = AskResponse(
    answer="Hello",
    citations=[
      Citation(
        research_id="r1",
        title="Attention",
        page=29,
        score=0.9,
        source_path="/data/researches/paper.pdf",
        filename="paper.pdf",
        display_name="Attention Is All You Need",
        authors=["Vaswani"],
      )
    ],
  )
  text = format_ask_response(response)
  assert "Attention Is All You Need, стр. 29 (Vaswani)" in text


def test_format_documents_list():
  from research_shared.domain.models import DocumentListItem, IngestStatus
  from vk_bot.handlers.bot_service import format_documents_list

  text = format_documents_list([
    DocumentListItem(
      research_id="r1",
      filename="paper_abc.pdf",
      display_name="Attention Is All You Need",
      status=IngestStatus.INDEXED,
      chunk_count=42,
    )
  ])
  assert "Attention Is All You Need" in text
  assert "42 чанков" in text


def test_format_ask_response_without_answer_dedupes_pages():
  chunks = [
    ResearchChunk(
      research_id="r1",
      title="Paper",
      text="First chunk text",
      source_path="/data/researches/paper.pdf",
      metadata={"page": 3},
    ),
    ResearchChunk(
      research_id="r1",
      title="Paper",
      text="Second chunk text",
      source_path="/data/researches/paper.pdf",
      metadata={"page": 3},
    ),
  ]
  citations = [
    Citation(
      research_id="r1",
      title="Paper",
      page=3,
      score=0.9,
      source_path="/data/researches/paper.pdf",
    ),
    Citation(
      research_id="r1",
      title="Paper",
      page=3,
      score=0.7,
      source_path="/data/researches/paper.pdf",
    ),
  ]
  response = AskResponse(context_chunks=chunks, citations=citations)
  text = format_ask_response(response)
  assert text.count("paper.pdf, стр. 3") == 2
  assert "First chunk text" in text or "Second chunk text" in text
