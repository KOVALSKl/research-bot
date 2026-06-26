from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vk_bot.config import VkBotSettings
from vk_bot.domain import IncomingMessage

DEFAULT_HELP_KEYWORDS = ("помощь", "/help", "?", "команды")


class Intent(Enum):
  GREETING = "greeting"
  HELP = "help"
  ASK = "ask"
  IDEA = "idea"
  LIST = "list"
  UPLOAD = "upload"
  UNKNOWN = "unknown"


@dataclass(frozen=True)
class RouteResult:
  intent: Intent
  ask_text: str = ""
  idea_text: str = ""


class CommandRouter:
  def __init__(
    self,
    settings: VkBotSettings,
    *,
    help_keywords: tuple[str, ...] = DEFAULT_HELP_KEYWORDS,
  ) -> None:
    self._greeting_keywords = tuple(
      keyword.lower() for keyword in settings.vk_greeting_keywords
    )
    self._ask_prefixes = sorted(
      settings.vk_ask_command_prefixes,
      key=len,
      reverse=True,
    )
    self._idea_prefixes = sorted(
      settings.vk_idea_command_prefixes,
      key=len,
      reverse=True,
    )
    self._list_prefixes = sorted(
      settings.vk_list_command_prefixes,
      key=len,
      reverse=True,
    )
    self._help_keywords = tuple(keyword.lower() for keyword in help_keywords)

  def resolve(self, message: IncomingMessage) -> RouteResult:
    pdf_attachments = [attachment for attachment in message.attachments if attachment.ext == "pdf"]
    if pdf_attachments:
      return RouteResult(Intent.UPLOAD)

    text = message.text.strip()
    normalized = text.lower()

    for keyword in self._greeting_keywords:
      if normalized == keyword:
        return RouteResult(Intent.GREETING)

    for keyword in self._help_keywords:
      if normalized == keyword:
        return RouteResult(Intent.HELP)

    for prefix in self._list_prefixes:
      prefix_lower = prefix.lower()
      if normalized == prefix_lower or normalized.startswith(f"{prefix_lower} "):
        return RouteResult(Intent.LIST)

    for prefix in self._idea_prefixes:
      prefix_lower = prefix.lower()
      if normalized == prefix_lower:
        return RouteResult(Intent.HELP)
      if normalized.startswith(prefix_lower):
        idea_text = text[len(prefix) :].strip()
        if idea_text.startswith(":"):
          idea_text = idea_text[1:].strip()
        if idea_text:
          return RouteResult(Intent.IDEA, idea_text=idea_text)

    for prefix in self._ask_prefixes:
      prefix_lower = prefix.lower()
      if prefix_lower == "?":
        if normalized.startswith("?") and len(text) > 1:
          ask_text = text[1:].strip()
          if ask_text:
            return RouteResult(Intent.ASK, ask_text=ask_text)
        continue

      if normalized.startswith(prefix_lower):
        ask_text = text[len(prefix) :].strip()
        if ask_text.startswith(":"):
          ask_text = ask_text[1:].strip()
        if ask_text:
          return RouteResult(Intent.ASK, ask_text=ask_text)

    return RouteResult(Intent.UNKNOWN)
