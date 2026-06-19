from __future__ import annotations

from pathlib import Path

from research_shared.domain.models import AskResponse, Citation, DocumentListItem, ResearchChunk
from research_shared.rag.citations import citation_display_name, citation_filename, dedupe_citations

from vk_bot.config import VkBotSettings
from vk_bot.core_api.client import CoreApiClient, CoreApiError
from vk_bot.domain import Attachment, IncomingMessage
from vk_bot.handlers import messages as ux
from vk_bot.handlers.router import CommandRouter, Intent
from vk_bot.security.rate_limiter import RateLimiter
from vk_bot.security.sanitizer import MessageSanitizer
from vk_bot.state.message_dedup import MessageDedupStore
from vk_bot.state.upload_naming import UploadNamingStore, UploadNamingSessionData
from vk_bot.state.user_queue import UserUploadQueueStore
from vk_bot.state.user_session import UserSessionStore
from vk_bot.vk.api import VkApiClientProtocol

from research_shared.logging_config import get_logger

logger = get_logger(__name__)

VK_MESSAGE_LIMIT = 4096
_CANCEL_COMMANDS = frozenset({"/cancel", "отмена"})


def _split_text(text: str, limit: int = VK_MESSAGE_LIMIT) -> list[str]:
  if len(text) <= limit:
    return [text]
  parts: list[str] = []
  remaining = text
  while remaining:
    if len(remaining) <= limit:
      parts.append(remaining)
      break
    split_at = remaining.rfind("\n", 0, limit)
    if split_at <= 0:
      split_at = limit
    parts.append(remaining[:split_at])
    remaining = remaining[split_at:].lstrip("\n")
  return parts


def _format_citation(citation: Citation) -> str:
  name = citation_display_name(citation)
  page = f", стр. {citation.page}" if citation.page is not None else ""
  authors = ", ".join(citation.authors) if citation.authors else "—"
  return f"• {name}{page} ({authors})"


def _unique_fragments(
  context_chunks: list[ResearchChunk],
  citations: list[Citation],
  *,
  limit: int = 5,
) -> list[tuple[str, str]]:
  """Return up to ``limit`` unique (label, excerpt) pairs for LLM-off mode."""
  deduped = dedupe_citations(citations) if citations else []
  if deduped:
    fragments: list[tuple[str, str]] = []
    chunk_by_key: dict[tuple[str, int | None, str], ResearchChunk] = {}
    for chunk in context_chunks:
      page = chunk.metadata.get("page")
      filename = chunk.source_path and chunk.source_path.split("/")[-1] or chunk.title
      key = (chunk.research_id, page, chunk.display_name or filename)
      if key not in chunk_by_key:
        chunk_by_key[key] = chunk
    for citation in deduped:
      key = (citation.research_id, citation.page, citation_display_name(citation))
      chunk = chunk_by_key.get(key)
      if chunk is None:
        continue
      page = citation.page
      page_str = f", стр. {page}" if page is not None else ""
      label = f"{citation_display_name(citation)}{page_str}"
      fragments.append((label, chunk.text[:300]))
      if len(fragments) >= limit:
        break
    return fragments

  seen: set[tuple[str, int | None, str]] = set()
  fragments = []
  for chunk in context_chunks:
    page = chunk.metadata.get("page")
    filename = chunk.source_path.split("/")[-1] if chunk.source_path else chunk.title
    label_name = chunk.display_name or filename
    key = (chunk.research_id, page, label_name)
    if key in seen:
      continue
    seen.add(key)
    page_str = f", стр. {page}" if page is not None else ""
    label = f"{label_name}{page_str}"
    fragments.append((label, chunk.text[:300]))
    if len(fragments) >= limit:
      break
  return fragments


def format_ask_response(response: AskResponse) -> str:
  if response.answer:
    body = response.answer
  else:
    lines = ["Найдены релевантные фрагменты:"]
    for label, excerpt in _unique_fragments(response.context_chunks, response.citations):
      lines.append(f"• {label}: {excerpt}…")
    body = "\n".join(lines)

  if response.citations:
    citations = dedupe_citations(response.citations)
    body += "\n\nИсточники:\n" + "\n".join(
      _format_citation(c) for c in citations
    )
  return body


def format_documents_list(items: list[DocumentListItem]) -> str:
  if not items:
    return ux.documents_list_empty()

  lines = [ux.format_documents_list_header(len(items))]
  for item in items:
    name = item.display_name or item.filename
    status = ux._STATUS_LABELS.get(item.status.value, item.status.value)
    chunk_part = f", {item.chunk_count} чанков" if item.chunk_count else ""
    lines.append(f"• {name} — {status}{chunk_part}")
  return "\n".join(lines)


class BotService:
  def __init__(
    self,
    settings: VkBotSettings,
    vk_api: VkApiClientProtocol,
    core_api: CoreApiClient,
    rate_limiter: RateLimiter,
    sanitizer: MessageSanitizer,
    queue_store: UserUploadQueueStore,
    router: CommandRouter,
    session_store: UserSessionStore,
    naming_store: UploadNamingStore,
    dedup_store: MessageDedupStore | None = None,
  ) -> None:
    self._settings = settings
    self._vk_api = vk_api
    self._core_api = core_api
    self._rate_limiter = rate_limiter
    self._sanitizer = sanitizer
    self._queue = queue_store
    self._router = router
    self._session = session_store
    self._naming = naming_store
    self._dedup = dedup_store

  @staticmethod
  def _is_cancel_command(text: str) -> bool:
    return text.strip().lower() in _CANCEL_COMMANDS

  async def handle(self, message: IncomingMessage) -> None:
    if message.is_outgoing:
      logger.debug(
        "Skipping outgoing message",
        extra={"user_id": message.user_id, "event": "message.skip_outgoing"},
      )
      return

    if self._dedup is not None and not await self._dedup.try_acquire(message):
      logger.debug(
        "Skipping duplicate message",
        extra={
          "user_id": message.user_id,
          "event": "message.skip_duplicate",
          "count": message.message_id,
        },
      )
      return

    route = self._router.resolve(message)
    pdf_attachments = [attachment for attachment in message.attachments if attachment.ext == "pdf"]
    sanitized_text = self._sanitizer.sanitize(message.text)

    if await self._naming.active(message.user_id):
      if self._is_cancel_command(message.text):
        await self._cancel_naming(message)
        return
      if route.intent in {Intent.LIST, Intent.ASK}:
        pass
      elif pdf_attachments:
        await self._vk_api.send_message(message.peer_id, ux.naming_session_busy())
        return
      elif sanitized_text and route.intent not in {Intent.GREETING, Intent.HELP}:
        await self._handle_naming_reply(message, sanitized_text)
        return

    logger.info(
      "Incoming message",
      extra={
        "user_id": message.user_id,
        "peer_id": message.peer_id,
        "intent": route.intent.value,
        "count": len(message.attachments),
        "event": "user.message",
      },
    )

    if not await self._rate_limiter.allow(
      message.user_id,
      message.text,
      attachment_count=len(pdf_attachments),
    ):
      if await self._rate_limiter.should_notify_rate_limit(message.user_id):
        await self._vk_api.send_message(message.peer_id, ux.rate_limited())
        await self._rate_limiter.mark_rate_limit_notified(message.user_id)
      return

    non_pdf_attachments = [
      attachment for attachment in message.attachments if attachment.ext != "pdf"
    ]

    if pdf_attachments:
      if not await self._session.has_seen(message.user_id):
        await self._vk_api.send_message(message.peer_id, ux.welcome())
        await self._session.mark_seen(message.user_id)
      await self._handle_pdf_upload(message, pdf_attachments)
      return

    if non_pdf_attachments and not sanitized_text:
      if not await self._session.has_seen(message.user_id):
        await self._vk_api.send_message(message.peer_id, ux.welcome())
        await self._session.mark_seen(message.user_id)
      await self._vk_api.send_message(message.peer_id, ux.unsupported_attachment())
      return

    first_visit = not await self._session.has_seen(message.user_id)
    if first_visit:
      await self._vk_api.send_message(message.peer_id, ux.welcome())
      await self._session.mark_seen(message.user_id)
      if route.intent in {Intent.GREETING, Intent.UNKNOWN}:
        return

    if route.intent == Intent.ASK:
      await self._handle_question(message, route.ask_text)
      return

    if route.intent == Intent.LIST:
      await self._handle_list(message)
      return

    if route.intent == Intent.GREETING:
      await self._vk_api.send_message(message.peer_id, ux.welcome())
      return

    if route.intent == Intent.HELP:
      await self._vk_api.send_message(message.peer_id, ux.help_text())
      return

    await self._vk_api.send_message(message.peer_id, ux.unknown_command())

  async def _handle_question(self, message: IncomingMessage, text: str) -> None:
    sanitized = self._sanitizer.sanitize(text)
    if not sanitized:
      await self._vk_api.send_message(message.peer_id, ux.unknown_command())
      return

    if len(sanitized) < self._settings.vk_min_question_length:
      await self._vk_api.send_message(
        message.peer_id,
        ux.question_too_short(self._settings.vk_min_question_length),
      )
      return

    await self._vk_api.send_message(message.peer_id, ux.processing_question())
    try:
      response = await self._core_api.ask(sanitized)
    except CoreApiError as exc:
      logger.exception("Ask failed for user %s", message.user_id)
      await self._vk_api.send_message(message.peer_id, ux.api_error(str(exc)))
      return

    formatted = format_ask_response(response)
    parts = _split_text(formatted)
    for index, part in enumerate(parts):
      attachment = None
      if (
        index == len(parts) - 1
        and self._settings.vk_ask_attach_enabled
        and response.source_files
      ):
        attachment = await self._build_source_attachment(message.peer_id, response)
      await self._vk_api.send_message(message.peer_id, part, attachment=attachment)

  async def _build_source_attachment(
    self,
    peer_id: int,
    response: AskResponse,
  ) -> str | None:
    max_attachments = self._settings.vk_ask_max_attachments
    for source_file in response.source_files[:max_attachments]:
      try:
        content, filename = await self._core_api.download_source_file(
          source_file.research_id,
        )
      except CoreApiError:
        logger.exception(
          "Source file download failed",
          extra={
            "research_id": source_file.research_id,
            "event": "ask.source_download_failed",
          },
        )
        continue

      upload_name = source_file.display_name or source_file.filename or filename
      try:
        return await self._vk_api.upload_doc_to_vk(
          content,
          upload_name,
          peer_id,
        )
      except Exception:
        logger.exception(
          "VK doc upload failed",
          extra={
            "attachment_name": upload_name,
            "event": "ask.doc_upload_failed",
          },
        )
    return None

  async def _handle_list(self, message: IncomingMessage) -> None:
    try:
      items = await self._core_api.list_documents()
    except CoreApiError as exc:
      logger.exception(
        "List documents failed",
        extra={"user_id": message.user_id, "event": "list.failed"},
      )
      await self._vk_api.send_message(message.peer_id, ux.api_error(str(exc)))
      return

    formatted = format_documents_list(items)
    for part in _split_text(formatted):
      await self._vk_api.send_message(message.peer_id, part)

  async def _cancel_naming(self, message: IncomingMessage) -> None:
    await self._naming.clear(message.user_id)
    await self._vk_api.send_message(message.peer_id, ux.naming_cancelled())

  async def _handle_naming_reply(self, message: IncomingMessage, text: str) -> None:
    session = await self._naming.get(message.user_id)
    if session is None:
      await self._vk_api.send_message(message.peer_id, ux.unknown_command())
      return

    current_file = session.files[session.current_index]
    display_name = current_file.original_name if text.strip() == "-" else text.strip()
    if not display_name:
      display_name = current_file.original_name

    await self._naming.append_name(message.user_id, display_name)
    session = await self._naming.advance(message.user_id)
    if session is None:
      return

    if session.current_index < len(session.files):
      next_file = session.files[session.current_index]
      await self._vk_api.send_message(
        message.peer_id,
        ux.ask_display_name(
          next_file.original_name,
          session.current_index + 1,
          len(session.files),
        ),
      )
      return

    await self._finalize_upload(message.user_id, session)

  async def _finalize_upload(self, user_id: int, session: UploadNamingSessionData) -> None:
    peer_id = session.peer_id
    task_ids: list[str] = []
    filenames: list[str] = []

    try:
      for index, file in enumerate(session.files):
        content = Path(file.temp_path).read_bytes()
        display_name = session.names[index] if index < len(session.names) else file.original_name
        result = await self._core_api.upload_document(
          content,
          file.original_name,
          display_name=display_name,
        )
        task_ids.append(str(result["task_id"]))
        filenames.append(display_name)
    except CoreApiError as exc:
      logger.exception(
        "Upload failed during naming finalize",
        extra={"user_id": user_id, "event": "upload.api_failed"},
      )
      await self._naming.clear(user_id)
      await self._vk_api.send_message(peer_id, ux.api_error(str(exc)))
      return

    await self._naming.clear(user_id)

    if not task_ids:
      await self._vk_api.send_message(peer_id, ux.api_error("Не удалось поставить задачи в очередь."))
      return

    await self._queue.save_batch(
      user_id,
      task_ids=task_ids,
      filenames=filenames,
      peer_id=peer_id,
    )
    logger.info(
      "Upload queued after naming wizard",
      extra={
        "user_id": user_id,
        "count": len(task_ids),
        "event": "upload.queued",
      },
    )
    await self._vk_api.send_message(peer_id, ux.tasks_queued())

  async def _handle_pdf_upload(
    self,
    message: IncomingMessage,
    attachments: list[Attachment],
  ) -> None:
    max_count = self._settings.vk_max_pdf_attachments
    if len(attachments) > max_count:
      logger.warning(
        "Too many PDF attachments",
        extra={
          "user_id": message.user_id,
          "count": len(attachments),
          "event": "upload.too_many_attachments",
        },
      )
      await self._vk_api.send_message(message.peer_id, ux.too_many_attachments(max_count))
      return

    if await self._queue.is_busy(message.user_id):
      await self._vk_api.send_message(message.peer_id, ux.queue_busy())
      return

    logger.info(
      "Starting PDF upload wizard",
      extra={
        "user_id": message.user_id,
        "count": len(attachments),
        "event": "upload.start",
      },
    )
    await self._vk_api.send_message(message.peer_id, ux.processing_upload())
    try:
      files = await self._vk_api.download_attachments(attachments)
    except Exception:
      logger.exception(
        "Attachment download failed",
        extra={"user_id": message.user_id, "event": "upload.download_failed"},
      )
      await self._vk_api.send_message(message.peer_id, ux.pdf_download_failed())
      return

    if not files:
      logger.error(
        "No attachments downloaded",
        extra={"user_id": message.user_id, "event": "upload.download_empty"},
      )
      await self._vk_api.send_message(message.peer_id, ux.pdf_download_failed())
      return

    naming_files = []
    for content, filename in files:
      naming_files.append(
        self._naming.save_temp_file(message.user_id, content, filename)
      )

    await self._naming.save(
      message.user_id,
      peer_id=message.peer_id,
      files=naming_files,
    )
    first_file = naming_files[0]
    await self._vk_api.send_message(
      message.peer_id,
      ux.ask_display_name(first_file.original_name, 1, len(naming_files)),
    )
