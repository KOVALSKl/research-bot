from pathlib import Path

import pytest
from research_shared.agents.models import AgentAskResponse, AgentSources, EvidenceItem, IdeaAssessment, RelevanceAssessment, RelevanceCriterion
from research_shared.domain.models import AskResponse, Citation, ExternalSourceFileRef, ResearchChunk, SourceFileRef
from research_shared.literature.models import ExternalPaper

from vk_bot.config import VkBotSettings
from vk_bot.core_api.client import CoreApiError
from vk_bot.domain import Attachment, IncomingMessage
from vk_bot.handlers.bot_service import BotService, format_agent_response, format_ask_response
from vk_bot.handlers import messages as ux
from vk_bot.handlers.router import CommandRouter
from vk_bot.security.rate_limiter import MemoryRateLimiter
from vk_bot.security.sanitizer import MessageSanitizer
from vk_bot.state.message_dedup import MemoryMessageDedupStore
from vk_bot.state.upload_naming import MemoryUploadNamingStore, NamingFile, UploadNamingSessionData
from vk_bot.state.user_session import MemoryUserSessionStore


def _sample_relevance(text: str = "Идея релевантна [1].") -> RelevanceAssessment:
  return RelevanceAssessment(
    level="medium",
    criteria=[
      RelevanceCriterion(
        name="topic_overlap",
        level="medium",
        detail=text,
      )
    ],
    rationale=text,
  )


class MockVkApi:
  def __init__(self) -> None:
    self.sent: list[tuple[int, str, str | None]] = []
    self.deleted: list[tuple[int, int]] = []
    self.downloaded: list[tuple[bytes, str]] = []
    self.download_raises: Exception | None = None
    self.upload_doc_result = "doc1_2"
    self.upload_doc_raises: Exception | None = None
    self._next_message_id = 100

  async def send_message(
    self,
    peer_id: int,
    text: str,
    *,
    attachment: str | None = None,
  ) -> int:
    self.sent.append((peer_id, text, attachment))
    self._next_message_id += 1
    return self._next_message_id

  async def delete_message(
    self,
    peer_id: int,
    message_id: int,
    *,
    delete_for_all: bool = True,
  ) -> None:
    self.deleted.append((peer_id, message_id))

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
    self.agent_ask_calls: list[tuple[str, str]] = []
    self.agent_ask_stream_calls: list[tuple[str, str]] = []
    self.upload_calls: list[tuple[bytes, str, str | None]] = []
    self.batch_calls: list[tuple[list[tuple[bytes, str]], list[str] | None]] = []
    self.download_calls: list[str] = []
    self.download_external_calls: list[tuple[str, str | None]] = []
    self.list_response: list = []
    self.agent_ask_response = AgentAskResponse(
      answer="Test answer",
      sources=AgentSources(),
      steps=[],
    )
    self.upload_response = {"task_id": "task-1", "research_id": "res-1"}
    self.batch_response = {
      "jobs": [
        {"task_id": "task-1", "filename": "a.pdf"},
        {"task_id": "task-2", "filename": "b.pdf"},
      ]
    }
    self.download_response = (b"%PDF", "paper.pdf")
    self.raise_on_agent_ask: Exception | None = None
    self.raise_on_download: Exception | None = None
    self.raise_on_list: Exception | None = None
    self.raise_on_upload: Exception | None = None

  async def agent_ask(
    self,
    message: str,
    mode: str = "question",
    limit: int | None = None,
  ) -> AgentAskResponse:
    if self.raise_on_agent_ask:
      raise self.raise_on_agent_ask
    self.agent_ask_calls.append((message, mode))
    return self.agent_ask_response

  async def agent_ask_stream(
    self,
    message: str,
    mode: str = "question",
    limit: int | None = None,
    *,
    on_progress=None,
    on_reasoning=None,
  ) -> AgentAskResponse:
    if self.raise_on_agent_ask:
      raise self.raise_on_agent_ask
    self.agent_ask_stream_calls.append((message, mode))
    if on_progress is not None:
      from research_shared.agents.models import AgentProgressEvent, AgentProgressStage

      for stage, stage_index, progress_message in (
        (AgentProgressStage.CLASSIFY, 1, "Определяю тип запроса…"),
        (AgentProgressStage.SYNTHESIZE, 2, "Формирую ответ…"),
      ):
        event = AgentProgressEvent(
          stage=stage,
          stage_index=stage_index,
          stage_total=2,
          message=progress_message,
        )
        result = on_progress(event)
        if hasattr(result, "__await__"):
          await result
    if on_reasoning is not None:
      from research_shared.agents.models import AgentReasoningEvent

      event = AgentReasoningEvent(
        iteration=1,
        max_iterations=6,
        thought="Searching locally for relevant fragments",
        action="local_hybrid_search",
        action_summary="Найдено 2 фрагмента",
      )
      result = on_reasoning(event)
      if hasattr(result, "__await__"):
        await result
    return self.agent_ask_response

  async def upload_document(
    self,
    content: bytes,
    filename: str,
    *,
    display_name: str | None = None,
  ) -> dict:
    if self.raise_on_upload:
      raise self.raise_on_upload
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
    if self.raise_on_list:
      raise self.raise_on_list
    return self.list_response

  async def get_task_status(self, task_id: str) -> dict:
    return {"status": "indexed", "task_id": task_id}

  async def download_source_file(self, research_id: str) -> tuple[bytes, str]:
    if self.raise_on_download:
      raise self.raise_on_download
    self.download_calls.append(research_id)
    return self.download_response

  async def download_external_pdf(
    self,
    cache_key: str,
    *,
    pdf_url: str | None = None,
  ) -> tuple[bytes, str]:
    if self.raise_on_download:
      raise self.raise_on_download
    self.download_external_calls.append((cache_key, pdf_url))
    return b"%PDF-external", "external.pdf"


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

  assert core.agent_ask_stream_calls == [("What is quantum computing?", "question")]
  assert any("💭" in text for _, text, _ in vk.sent)
  assert any("Test answer" in text for _, text, _ in vk.sent)
  assert vk.deleted


@pytest.mark.asyncio
async def test_ask_attaches_source_pdf(bot_service):
  service, vk, core, _ = bot_service
  core.agent_ask_response = AgentAskResponse(
    answer="Answer",
    sources=AgentSources(
      local=[
        Citation(
          research_id="r1",
          title="Paper",
          page=2,
          score=0.9,
          source_path="/data/researches/paper.pdf",
          filename="paper.pdf",
        )
      ],
      local_indices=[1],
    ),
    source_files=[
      SourceFileRef(
        research_id="r1",
        filename="paper.pdf",
        display_name="paper.pdf",
        path="/data/researches/paper.pdf",
      )
    ],
    steps=[],
  )

  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="/ask What is quantum computing?"),
  )

  assert core.download_calls == ["r1"]
  assert any(attachment == "doc1_2" for _, _, attachment in vk.sent)


@pytest.mark.asyncio
async def test_ask_attaches_external_source_pdf(bot_service):
  service, vk, core, _ = bot_service
  core.agent_ask_response = AgentAskResponse(
    answer="Answer citing [E1]",
    sources=AgentSources(
      external=[
        ExternalPaper(
          title="External Paper",
          url="https://example.org/paper",
          pdf_url="https://example.org/paper.pdf",
          source="openalex",
          abstract="Abstract",
        )
      ],
      external_indices=[1],
    ),
    external_source_files=[
      ExternalSourceFileRef(
        external_index=1,
        title="External Paper",
        cache_key="cache123",
        filename="External_Paper.pdf",
        pdf_url="https://example.org/paper.pdf",
        display_name="External Paper",
      )
    ],
    steps=[],
  )

  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="/ask What is quantum computing?"),
  )

  assert core.download_external_calls == [("cache123", "https://example.org/paper.pdf")]
  assert any(attachment == "doc1_2" for _, _, attachment in vk.sent)
  assert any("📎 PDF приложен" in text for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_idea_command_calls_agent_with_idea_mode(bot_service):
  service, vk, core, _ = bot_service
  core.agent_ask_response = AgentAskResponse(
    answer="Оценка идеи [E1]",
    sources=AgentSources(external_indices=[1]),
    idea_assessment=IdeaAssessment(
      relevance=_sample_relevance("Релевантно [E1]."),
      evidence_for=[EvidenceItem(text="Поддержка [E1].", source_type="external")],
      evidence_against=[EvidenceItem(text="Риски [E1].", source_type="external")],
      success_outlook="Умеренные перспективы [E1].",
      confidence="medium",
      summary="Итог [E1].",
    ),
    steps=[],
  )

  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="/idea GNN для fraud detection"),
  )

  assert core.agent_ask_stream_calls == [("GNN для fraud detection", "idea_evaluation")]
  assert core.agent_ask_response.idea_assessment is not None
  assert any("Оценка идеи" in text for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_ask_attachment_failure_still_sends_text(bot_service):
  service, vk, core, _ = bot_service
  core.agent_ask_response = AgentAskResponse(
    answer="Answer",
    sources=AgentSources(
      local=[
        Citation(
          research_id="r1",
          title="Paper",
          page=1,
          score=0.9,
          source_path="/data/researches/paper.pdf",
          filename="paper.pdf",
        )
      ],
      local_indices=[1],
    ),
    source_files=[
      SourceFileRef(
        research_id="r1",
        filename="paper.pdf",
        display_name="paper.pdf",
        path="/data/researches/paper.pdf",
      )
    ],
    steps=[],
  )
  vk.upload_doc_raises = RuntimeError("upload failed")

  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="/ask What is quantum computing?"),
  )

  assert any("Answer" in text and "PDF: paper.pdf" in text for _, text, _ in vk.sent)
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

  assert core.agent_ask_stream_calls == []
  assert any("не понял" in text.lower() for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_short_ask_skips_core_api(bot_service):
  service, vk, core, _ = bot_service
  await service.handle(IncomingMessage(user_id=1, peer_id=1, text="/ask short"))

  assert core.agent_ask_stream_calls == []
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

  assert len(core.agent_ask_stream_calls) == 1
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

  assert core.agent_ask_stream_calls == []
  assert vk.sent == []


@pytest.mark.asyncio
async def test_first_visit_sends_welcome(bot_service):
  service, vk, core, _ = bot_service
  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="What is quantum computing?"),
  )

  assert core.agent_ask_stream_calls == []
  assert any("бот для поиска" in text.lower() for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_greeting_sends_welcome(bot_service):
  service, vk, core, _ = bot_service
  await service.handle(IncomingMessage(user_id=2, peer_id=2, text="привет"))

  assert core.agent_ask_stream_calls == []
  assert sum("бот для поиска" in text.lower() for _, text, _ in vk.sent) >= 1


@pytest.mark.asyncio
async def test_first_visit_help_sends_welcome_and_help(bot_service):
  service, vk, core, _ = bot_service
  await service.handle(IncomingMessage(user_id=7, peer_id=7, text="помощь"))

  assert core.agent_ask_stream_calls == []
  assert any("бот для поиска" in text.lower() for _, text, _ in vk.sent)
  assert any("команды бота" in text.lower() for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_duplicate_message_is_ignored(bot_service):
  service, vk, core, _ = bot_service
  message = IncomingMessage(user_id=8, peer_id=8, message_id=999, text="привет")

  await service.handle(message)
  await service.handle(message)

  assert core.agent_ask_stream_calls == []
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
  assert "• paper.pdf (Vaswani) — стр. 29" in text


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
  assert "Attention Is All You Need (Vaswani) — стр. 29" in text


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
  assert "• paper.pdf (—) — стр. 3" in text
  assert text.count("стр. 3") >= 2
  assert "First chunk text" in text or "Second chunk text" in text


def test_format_agent_response_local_only():
  response = AgentAskResponse(
    answer="Agent answer with [1].",
    sources=AgentSources(
      local=[
        Citation(
          research_id="r1",
          title="Paper",
          page=5,
          score=0.8,
          source_path="/data/researches/paper.pdf",
          filename="paper.pdf",
          authors=["Alice"],
        )
      ],
      local_indices=[1],
    ),
    steps=[],
  )
  text = format_agent_response(response)
  assert "Agent answer with [1]." in text
  assert "Локальные источники:" in text
  assert "[1] paper.pdf (Alice) — стр. 5" in text
  assert "Внешние публикации:" not in text


def test_format_agent_response_grouped_pages():
  response = AgentAskResponse(
    answer="Agent answer with [1] and [2].",
    sources=AgentSources(
      local=[
        Citation(
          research_id="r1",
          title="Paper",
          page=5,
          score=0.8,
          filename="paper.pdf",
          authors=["Alice"],
        ),
        Citation(
          research_id="r1",
          title="Paper",
          page=12,
          score=0.7,
          filename="paper.pdf",
          authors=["Alice"],
        ),
        Citation(
          research_id="r1",
          title="Paper",
          page=23,
          score=0.6,
          filename="paper.pdf",
          authors=["Alice"],
        ),
      ],
      local_indices=[1, 2, 3],
    ),
    steps=[],
  )
  text = format_agent_response(response)
  assert "[1] paper.pdf (Alice) — стр. 5, 12, 23" in text
  assert text.count("[1] paper.pdf") == 1


def test_agent_reasoning_empty_thought_uses_action_summary():
  text = ux.agent_reasoning("", "Найдено 3 фрагмента")
  assert text == "💭 Найдено 3 фрагмента"


def test_agent_reasoning_empty_thought_uses_action_label():
  text = ux.agent_reasoning("", None, action="local_hybrid_search")
  assert "PDF" in text


@pytest.mark.asyncio
async def test_ask_stream_progress_delete_replace(bot_service):
  service, vk, core, _ = bot_service

  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="/ask What is quantum computing?"),
  )

  assert len(vk.deleted) >= 1
  assert any("Определяю тип запроса" in text for _, text, _ in vk.sent)
  assert any("Формирую ответ" in text for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_ask_stream_react_reasoning_delete_replace(bot_service):
  service, vk, core, _ = bot_service

  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="/ask What is quantum computing?"),
  )

  assert len(vk.deleted) >= 1
  assert any("💭" in text for _, text, _ in vk.sent)
  assert any("Searching locally" in text for _, text, _ in vk.sent)
  assert any("Test answer" in text for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_ask_stream_deletes_status_before_final_answer(bot_service):
  service, vk, core, _ = bot_service
  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="/ask What is quantum computing?"),
  )

  assert len(vk.deleted) >= 1
  assert any("Test answer" in text for _, text, _ in vk.sent)


@pytest.mark.asyncio
async def test_ask_error_deletes_status(bot_service):
  service, vk, core, _ = bot_service
  core.raise_on_agent_ask = CoreApiError("stream failed")

  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="/ask What is quantum computing?"),
  )

  assert vk.deleted
  sent_text = " ".join(text for _, text, _ in vk.sent)
  assert "stream failed" not in sent_text
  assert "Сервис ответов" in sent_text


@pytest.mark.asyncio
async def test_ask_connection_error_user_message(bot_service):
  service, vk, core, _ = bot_service
  core.raise_on_agent_ask = CoreApiError("Agent stream request failed: ConnectError('refused')")

  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="/ask What is quantum computing?"),
  )

  sent_text = " ".join(text for _, text, _ in vk.sent)
  assert "ConnectError" not in sent_text
  assert "Сервис ответов временно недоступен" in sent_text


@pytest.mark.asyncio
async def test_list_documents_503_user_message(bot_service):
  service, vk, core, _ = bot_service
  core.raise_on_list = CoreApiError("List documents failed", status_code=503)

  await service.handle(
    IncomingMessage(user_id=1, peer_id=1, text="/list"),
  )

  sent_text = " ".join(text for _, text, _ in vk.sent)
  assert "503" not in sent_text
  assert "запускается" in sent_text.lower()


@pytest.mark.asyncio
async def test_upload_timeout_user_message(bot_service, tmp_path):
  service, vk, core, _ = bot_service
  core.raise_on_upload = CoreApiError("Upload failed: timed out")

  temp_file = tmp_path / "paper.pdf"
  temp_file.write_bytes(b"%PDF")
  session = UploadNamingSessionData(
    peer_id=1,
    files=[NamingFile(original_name="paper.pdf", temp_path=str(temp_file))],
    names=["Paper"],
    current_index=1,
  )

  await service._finalize_upload(1, session)

  sent_text = " ".join(text for _, text, _ in vk.sent)
  assert "timed out" not in sent_text.lower()
  assert "слишком много времени" in sent_text.lower()


def test_format_agent_response_with_external():
  response = AgentAskResponse(
    answer="Combined answer with [1] and [E1].",
    sources=AgentSources(
      local=[
        Citation(
          research_id="r1",
          title="Local",
          page=1,
          score=0.9,
          filename="local.pdf",
          authors=["Ann"],
        )
      ],
      external=[
        ExternalPaper(
          title="External Paper",
          authors=["Bob"],
          year=2024,
          abstract="Abstract text.",
          doi="10.1234/test",
          url="https://example.org/paper",
          source="openalex",
        )
      ],
      local_indices=[1],
      external_indices=[1],
    ),
    steps=[],
  )
  text = format_agent_response(response)
  assert "Combined answer with [1] and [E1]." in text
  assert "Локальные источники:" in text
  assert "Внешние публикации:" in text
  assert "[1] local.pdf (Ann) — стр. 1" in text
  assert "[E1] External Paper (2024) (Bob) — 10.1234/test" in text


def test_format_agent_response_with_idea_assessment():
  response = AgentAskResponse(
    mode="idea_evaluation",
    answer="Краткая оценка с [1].",
    idea_assessment=IdeaAssessment(
      relevance=_sample_relevance("Идея релевантна [1]."),
      evidence_for=[EvidenceItem(text="Поддержка [1].")],
      evidence_against=[EvidenceItem(text="Риск [E1].")],
      success_outlook="Умеренные перспективы [E1].",
      confidence="medium",
    ),
    sources=AgentSources(
      local=[
        Citation(
          research_id="r1",
          title="Paper",
          page=1,
          score=0.8,
          source_path="/data/paper.pdf",
          filename="paper.pdf",
          authors=["Alice"],
        )
      ],
      local_indices=[1],
    ),
    steps=[],
  )
  text = format_agent_response(response)
  assert "📊 Оценка идеи" in text
  assert "Релевантность:" in text
  assert "Критерии:" in text
  assert "Обоснование:" in text
  assert "Аргументы за:" in text
  assert "Аргументы против:" in text
  assert "Уверенность: средняя" in text
  assert "Локальные источники:" in text
