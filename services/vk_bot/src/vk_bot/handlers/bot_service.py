from __future__ import annotations

import re
from pathlib import Path

from research_shared.agents.models import (
  AgentAskResponse,
  AgentProgressEvent,
  AgentReasoningEvent,
  IdeaAssessment,
  RelevanceAssessment,
  RelevanceCriterion,
)
from research_shared.domain.models import (
  AskResponse,
  Citation,
  DocumentListItem,
  ExternalSourceFileRef,
  ResearchChunk,
  SourceFileRef,
)
from research_shared.literature.models import ExternalPaper
from research_shared.rag.citations import (
  CitationGroup,
  citation_display_name,
  citation_filename,
  dedupe_citations,
  group_citations_by_document,
)

from vk_bot.config import VkBotSettings
from vk_bot.core_api.client import CoreApiClient, CoreApiError
from vk_bot.core_api.errors import map_core_api_error
from vk_bot.domain import Attachment, IncomingMessage
from vk_bot.handlers import messages as ux
from vk_bot.handlers.router import CommandRouter, Intent
from vk_bot.security.rate_limiter import RateLimiter
from vk_bot.security.sanitizer import MessageSanitizer
from vk_bot.state.conversation import ConversationStore
from vk_bot.state.message_dedup import MessageDedupStore
from vk_bot.state.upload_naming import UploadNamingStore, UploadNamingSessionData
from vk_bot.state.user_queue import UserUploadQueueStore
from vk_bot.state.user_session import UserSessionStore
from vk_bot.vk.api import VkApiClientProtocol

from research_shared.logging_config import get_logger

logger = get_logger(__name__)

VK_MESSAGE_LIMIT = 4096
_CANCEL_COMMANDS = frozenset({"/cancel", "отмена"})
_INLINE_CITATION_PATTERN = re.compile(r"\[(?:E)?\d+\]")

_DISPLAY_MATH = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_DISPLAY_DOLLAR = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_DOLLAR = re.compile(r"\$([^\$\n]+?)\$")
_LATEX_SUBS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\\frac\{([^}]+)\}\{([^}]+)\}"), r"\1/\2"),
    (re.compile(r"\\sqrt\{([^}]+)\}"), r"√(\1)"),
    (re.compile(r"\\times"), "×"),
    (re.compile(r"\\cdot"), "·"),
    (re.compile(r"\\leq"), "≤"),
    (re.compile(r"\\geq"), "≥"),
    (re.compile(r"\\neq"), "≠"),
    (re.compile(r"\\approx"), "≈"),
    (re.compile(r"\\infty"), "∞"),
    (re.compile(r"\\alpha"), "α"),
    (re.compile(r"\\beta"), "β"),
    (re.compile(r"\\gamma"), "γ"),
    (re.compile(r"\\delta"), "δ"),
    (re.compile(r"\\sigma"), "σ"),
    (re.compile(r"\\mu"), "μ"),
    (re.compile(r"\\pi"), "π"),
    (re.compile(r"\\sum"), "∑"),
    (re.compile(r"\\prod"), "∏"),
    (re.compile(r"\\int"), "∫"),
    (re.compile(r"\\[a-zA-Z]+\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\[a-zA-Z]+"), ""),
    (re.compile(r"[{}]"), ""),
]


def _strip_latex(text: str) -> str:
  def _clean(expr: str) -> str:
    for pattern, repl in _LATEX_SUBS:
      expr = pattern.sub(repl, expr)
    return expr.strip()

  text = _DISPLAY_DOLLAR.sub(lambda m: f"\n{_clean(m.group(1))}\n", text)
  text = _DISPLAY_MATH.sub(lambda m: f"\n{_clean(m.group(1))}\n", text)
  text = _INLINE_DOLLAR.sub(lambda m: _clean(m.group(1)), text)
  text = _INLINE_MATH.sub(lambda m: _clean(m.group(1)), text)
  return text


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


def _format_citation_group(group: CitationGroup, index: int) -> str:
  authors = ", ".join(group.authors) if group.authors else "—"
  if group.pages:
    pages = ", ".join(str(page) for page in group.pages)
    page_part = f" — стр. {pages}"
  else:
    page_part = ""
  url_part = f"\n    {group.source_url}" if group.source_url else ""
  return f"{index}. {group.display_name} ({authors}){page_part}{url_part}"


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


_RELEVANCE_LEVEL_LABELS = {
  "low": "низкая",
  "medium": "средняя",
  "high": "высокая",
}

_CRITERION_LABELS = {
  "local_sources": "Локальные источники",
  "external_publications": "Внешние публикации",
  "topic_overlap": "Пересечение с темой",
}


def _format_relevance_assessment(relevance: RelevanceAssessment) -> list[str]:
  level_label = _RELEVANCE_LEVEL_LABELS.get(relevance.level, relevance.level)
  lines = [
    f"Релевантность: {level_label}",
    "",
    "Критерии:",
  ]
  for criterion in relevance.criteria:
    name_label = _CRITERION_LABELS.get(criterion.name, criterion.name)
    criterion_level = _RELEVANCE_LEVEL_LABELS.get(criterion.level, criterion.level)
    lines.append(f"• {name_label} ({criterion_level}): {criterion.detail}")
  if relevance.rationale.strip():
    lines.extend(["", f"Обоснование: {relevance.rationale}"])
  return lines


def _format_idea_assessment_block(assessment: IdeaAssessment) -> str:
  confidence_labels = _RELEVANCE_LEVEL_LABELS
  lines = ["📊 Оценка идеи", ""]
  lines.extend(_format_relevance_assessment(assessment.relevance))
  if assessment.evidence_for:
    lines.append("")
    lines.append("Аргументы за:")
    lines.extend(f"• {item.text}" for item in assessment.evidence_for)
  if assessment.evidence_against:
    lines.append("")
    lines.append("Аргументы против:")
    lines.extend(f"• {item.text}" for item in assessment.evidence_against)
  lines.extend(
    [
      "",
      f"Перспективы: {assessment.success_outlook}",
      f"Уверенность: {confidence_labels.get(assessment.confidence, assessment.confidence)}",
    ]
  )
  return "\n".join(lines)


def format_agent_response(
  response: AgentAskResponse,
  *,
  delivered_external_indices: set[int] | None = None,
) -> str:
  if response.idea_assessment is not None:
    body = _format_idea_assessment_block(response.idea_assessment)
    if response.answer.strip() and response.answer.strip() not in body:
      body = f"{response.answer.strip()}\n\n{body}"
  else:
    body = response.answer

  local_block = _format_local_sources_block(response)
  external_block = _format_external_sources_block(
    response,
    delivered_external_indices=delivered_external_indices,
  )

  if local_block:
    body += "\n\n" + local_block
  if external_block:
    body += "\n\n" + external_block

  if _INLINE_CITATION_PATTERN.search(response.answer):
    if not local_block and not external_block:
      body += (
        "\n\n⚠️ В ответе есть ссылки на источники, "
        "но список источников сейчас недоступен."
      )

  return body


def _format_local_sources_block(response: AgentAskResponse) -> str:
  if not response.sources.local:
    return ""

  indices = response.sources.local_indices
  if indices and len(indices) == len(response.sources.local):
    lines: list[str] = []
    grouped: dict[tuple[str, str], tuple[int, CitationGroup]] = {}
    group_order: list[tuple[str, str]] = []

    for index, citation in zip(indices, response.sources.local, strict=True):
      key = (citation.research_id, citation_display_name(citation))
      if key not in grouped:
        group = group_citations_by_document([citation])[0]
        grouped[key] = (index, group)
        group_order.append(key)
        continue

      first_index, group = grouped[key]
      page = citation.page
      if page is not None and page not in group.pages:
        group.pages.append(page)
        group.pages.sort()

    for key in group_order:
      index, group = grouped[key]
      lines.append(_format_indexed_local_group(index, group))

    return "Локальные источники:\n" + "\n\n".join(lines)

  groups = group_citations_by_document(dedupe_citations(response.sources.local))
  return "Локальные источники:\n" + "\n\n".join(
    _format_citation_group(g, i) for i, g in enumerate(groups, 1)
  )


def _format_external_sources_block(
  response: AgentAskResponse,
  *,
  delivered_external_indices: set[int] | None = None,
) -> str:
  if not response.sources.external:
    return ""

  indices = response.sources.external_indices
  if indices and len(indices) == len(response.sources.external):
    lines = [
      _format_indexed_external_paper(
        index,
        paper,
        pdf_attached=delivered_external_indices is not None and index in delivered_external_indices,
      )
      for index, paper in zip(indices, response.sources.external, strict=True)
    ]
  else:
    lines = [_format_external_paper(paper) for paper in response.sources.external]

  return "Внешние публикации:\n" + "\n\n".join(lines)


def _format_indexed_local_group(index: int, group: CitationGroup) -> str:
  authors = ", ".join(group.authors) if group.authors else "—"
  if group.pages:
    pages = ", ".join(str(page) for page in group.pages)
    page_part = f" — стр. {pages}"
  else:
    page_part = ""
  url_part = f"\n    {group.source_url}" if group.source_url else ""
  return f"[{index}] {group.display_name} ({authors}){page_part}{url_part}"


def _format_indexed_external_paper(
  index: int,
  paper: ExternalPaper,
  *,
  pdf_attached: bool = False,
) -> str:
  authors = ", ".join(paper.authors) if paper.authors else "—"
  year = f" ({paper.year})" if paper.year is not None else ""
  link = paper.doi or paper.url or "—"
  attachment_note = " — 📎 PDF приложен" if pdf_attached else ""
  return f"[E{index}] {paper.title}{year} ({authors}) — {link}{attachment_note}"


def _format_external_paper(paper: ExternalPaper) -> str:
  authors = ", ".join(paper.authors) if paper.authors else "—"
  year = f" ({paper.year})" if paper.year is not None else ""
  link = paper.doi or paper.url
  return f"• {paper.title}{year} ({authors}) — {link}"


def format_ask_response(response: AskResponse) -> str:
  if response.answer:
    body = response.answer
  else:
    lines = ["Найдены релевантные фрагменты:"]
    for label, excerpt in _unique_fragments(response.context_chunks, response.citations):
      lines.append(f"• {label}: {excerpt}…")
    body = "\n".join(lines)

  if response.citations:
    groups = group_citations_by_document(dedupe_citations(response.citations))
    body += "\n\nИсточники:\n" + "\n\n".join(
      _format_citation_group(g, i) for i, g in enumerate(groups, 1)
    )
  return body


def format_documents_list(items: list[DocumentListItem]) -> str:
  if not items:
    return ux.documents_list_empty()

  lines = [ux.format_documents_list_header(len(items))]
  for index, item in enumerate(items, 1):
    name = item.display_name or item.filename
    status = ux._STATUS_LABELS.get(item.status.value, item.status.value)
    chunk_part = f", {item.chunk_count} чанков" if item.chunk_count else ""
    line = f"{index}. {name} — {status}{chunk_part}"
    if item.source_url:
      line += f"\n   {item.source_url}"
    lines.append(line)
  return "\n\n".join(lines)


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
    conversation_store: ConversationStore | None = None,
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
    self._conversation = conversation_store

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
      await self._handle_question(message, route.ask_text, mode="question")
      return

    if route.intent == Intent.IDEA:
      await self._handle_question(message, route.idea_text, mode="idea_evaluation")
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

  async def _handle_question(
    self,
    message: IncomingMessage,
    text: str,
    *,
    mode: str = "auto",
  ) -> None:
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

    status_id: int | None = await self._vk_api.send_message(
      message.peer_id,
      ux.processing_question(),
    )

    async def _replace_status(text: str) -> None:
      nonlocal status_id
      if status_id is not None:
        await self._vk_api.delete_message(message.peer_id, status_id)
      status_id = await self._vk_api.send_message(message.peer_id, text)

    async def on_progress(event: AgentProgressEvent) -> None:
      await _replace_status(event.message)

    async def on_reasoning(event: AgentReasoningEvent) -> None:
      if not event.thought.strip() and not event.action_summary and not event.action:
        return
      reasoning_text = ux.agent_reasoning(
        event.thought,
        event.action_summary,
        action=event.action,
      )
      await _replace_status(reasoning_text)

    history = []
    if self._conversation is not None and self._settings.vk_conversation_history_enabled:
      history = await self._conversation.get(message.user_id)

    try:
      response = await self._core_api.agent_ask_stream(
        sanitized,
        mode=mode,
        conversation_history=history,
        on_progress=on_progress,
        on_reasoning=on_reasoning,
      )
    except CoreApiError as exc:
      logger.exception("Ask failed for user %s", message.user_id)
      if status_id is not None:
        await self._vk_api.delete_message(message.peer_id, status_id)
      await self._vk_api.send_message(message.peer_id, map_core_api_error(exc))
      return

    if status_id is not None:
      await self._vk_api.delete_message(message.peer_id, status_id)

    if self._conversation is not None and self._settings.vk_conversation_history_enabled:
      await self._conversation.append(message.user_id, sanitized, response.answer)

    attachment, fallbacks, delivered_external = await self._deliver_source_attachments(
      message.peer_id,
      response.source_files,
      response.external_source_files,
    )
    formatted = format_agent_response(
      response,
      delivered_external_indices=delivered_external,
    )
    if fallbacks:
      formatted += "\n\n" + "\n".join(fallbacks)
    formatted = _strip_latex(formatted)

    parts = _split_text(formatted)
    for index, part in enumerate(parts):
      part_attachment = None
      if (
        index == len(parts) - 1
        and self._settings.vk_ask_attach_enabled
        and attachment
      ):
        part_attachment = attachment
      await self._vk_api.send_message(message.peer_id, part, attachment=part_attachment)

  async def _deliver_source_attachments(
    self,
    peer_id: int,
    source_files: list[SourceFileRef],
    external_source_files: list[ExternalSourceFileRef] | None = None,
  ) -> tuple[str | None, list[str], set[int]]:
    if not self._settings.vk_ask_attach_enabled:
      return None, [], set()

    max_attachments = self._settings.vk_ask_max_attachments
    attachments: list[str] = []
    fallbacks: list[str] = []
    delivered_external: set[int] = set()
    remaining = max_attachments

    for source_file in source_files:
      if remaining <= 0:
        break
      display_name = (
        source_file.display_name
        or source_file.filename
        or source_file.research_id
      )
      try:
        content, filename = await self._core_api.download_source_file(
          source_file.research_id,
        )
      except CoreApiError:
        logger.exception(
          "Source file download failed",
          extra={
            "research_id": source_file.research_id,
            "event": "ask.source_delivery_failed",
          },
        )
        continue

      upload_name = source_file.display_name or source_file.filename or filename
      try:
        attachment = await self._vk_api.upload_doc_to_vk(
          content,
          upload_name,
          peer_id,
        )
        attachments.append(attachment)
        remaining -= 1
      except Exception:
        logger.exception(
          "VK doc upload failed",
          extra={
            "attachment_name": upload_name,
            "event": "ask.source_delivery_failed",
          },
        )

    for external_file in external_source_files or []:
      if remaining <= 0:
        break
      display_name = external_file.display_name or external_file.title
      try:
        content, filename = await self._core_api.download_external_pdf(
          external_file.cache_key,
          pdf_url=external_file.pdf_url,
        )
      except CoreApiError:
        logger.exception(
          "External PDF download failed",
          extra={
            "cache_key": external_file.cache_key,
            "event": "external_pdf.fetch_failed",
          },
        )
        continue

      upload_name = external_file.display_name or external_file.filename or filename
      try:
        attachment = await self._vk_api.upload_doc_to_vk(
          content,
          upload_name,
          peer_id,
        )
        attachments.append(attachment)
        delivered_external.add(external_file.external_index)
        remaining -= 1
      except Exception:
        logger.exception(
          "VK external doc upload failed",
          extra={
            "attachment_name": upload_name,
            "event": "ask.source_delivery_failed",
          },
        )

    if not attachments and not fallbacks:
      return None, [], delivered_external

    joined = ",".join(attachments) if attachments else None
    return joined, fallbacks, delivered_external

  async def _handle_list(self, message: IncomingMessage) -> None:
    try:
      items = await self._core_api.list_documents()
    except CoreApiError as exc:
      logger.exception(
        "List documents failed",
        extra={"user_id": message.user_id, "event": "list.failed"},
      )
      await self._vk_api.send_message(message.peer_id, map_core_api_error(exc))
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
      await self._vk_api.send_message(peer_id, map_core_api_error(exc))
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
