import json
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _parse_csv_list(value: object) -> object:
  if isinstance(value, str):
    stripped = value.strip()
    if not stripped:
      return []
    if stripped.startswith("["):
      try:
        parsed = json.loads(stripped)
      except json.JSONDecodeError:
        parsed = None
      if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in stripped.split(",") if item.strip()]
  return value


CsvList = Annotated[list[str], NoDecode]


class VkBotSettings(BaseSettings):
  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    populate_by_name=True,
  )

  vk_bot_token: str = Field(default="", alias="VK_BOT_TOKEN")
  vk_group_id: int = Field(default=0, alias="VK_GROUP_ID")
  vk_api_version: str = Field(default="5.199", alias="VK_API_VERSION")
  vk_transport: str = Field(default="long_polling", alias="VK_TRANSPORT")

  vk_callback_secret: str = Field(default="", alias="VK_CALLBACK_SECRET")
  vk_callback_confirmation: str = Field(default="", alias="VK_CALLBACK_CONFIRMATION")
  vk_callback_host: str = Field(default="0.0.0.0", alias="VK_CALLBACK_HOST")
  vk_callback_port: int = Field(default=8081, alias="VK_CALLBACK_PORT")
  vk_callback_path: str = Field(default="/vk/callback", alias="VK_CALLBACK_PATH")

  core_api_base_url: str = Field(default="http://localhost:8000", alias="CORE_API_BASE_URL")
  redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
  vk_bot_redis_url: str = Field(default="", alias="VK_BOT_REDIS_URL")
  vk_bot_redis_key_prefix: str = Field(default="vk_bot:", alias="VK_BOT_REDIS_KEY_PREFIX")

  vk_rate_limit_backend: str = Field(default="redis", alias="VK_RATE_LIMIT_BACKEND")
  vk_rate_limit_max_messages: int = Field(default=5, alias="VK_RATE_LIMIT_MAX_MESSAGES")
  vk_rate_limit_window_seconds: int = Field(default=60, alias="VK_RATE_LIMIT_WINDOW_SECONDS")
  vk_debounce_seconds: int = Field(default=2, alias="VK_DEBOUNCE_SECONDS")

  vk_min_question_length: int = Field(default=12, alias="VK_MIN_QUESTION_LENGTH")
  vk_max_question_length: int = Field(default=2000, alias="VK_MAX_QUESTION_LENGTH")
  vk_ask_default_limit: int = Field(default=10, alias="VK_ASK_DEFAULT_LIMIT")

  vk_user_batch_ttl_seconds: int = Field(default=86400, alias="VK_USER_BATCH_TTL_SECONDS")
  vk_batch_poll_interval_seconds: int = Field(default=15, alias="VK_BATCH_POLL_INTERVAL_SECONDS")

  vk_core_api_timeout_seconds: float = Field(default=30.0, alias="VK_CORE_API_TIMEOUT_SECONDS")
  vk_core_api_startup_wait_seconds: float = Field(
    default=120.0,
    alias="VK_CORE_API_STARTUP_WAIT_SECONDS",
  )
  vk_core_api_startup_poll_interval_seconds: float = Field(
    default=2.0,
    alias="VK_CORE_API_STARTUP_POLL_INTERVAL_SECONDS",
  )
  vk_core_api_retry_max: int = Field(
    default=3,
    validation_alias=AliasChoices(
      "VK_CORE_API_RETRY_MAX",
      "VK_CORE_API_RETRY_MAX_ATTEMPTS",
    ),
  )
  vk_core_api_retry_backoff_seconds: float = Field(
    default=1.0,
    validation_alias=AliasChoices(
      "VK_CORE_API_RETRY_BACKOFF_SECONDS",
      "VK_CORE_API_RETRY_BASE_DELAY_SECONDS",
    ),
  )
  vk_ask_timeout_seconds: float = Field(default=180.0, alias="VK_ASK_TIMEOUT_SECONDS")
  vk_attachment_max_bytes: int = Field(default=52_428_800, alias="VK_ATTACHMENT_MAX_BYTES")

  vk_message_dedup_ttl_seconds: int = Field(default=86400, alias="VK_MESSAGE_DEDUP_TTL_SECONDS")
  vk_message_enrich_enabled: bool = Field(default=True, alias="VK_MESSAGE_ENRICH_ENABLED")
  vk_docs_resolve_url: bool = Field(default=True, alias="VK_DOCS_RESOLVE_URL")

  vk_max_pdf_attachments: int = Field(default=3, alias="VK_MAX_PDF_ATTACHMENTS")
  vk_ask_max_attachments: int = Field(default=1, alias="VK_ASK_MAX_ATTACHMENTS")
  vk_ask_attach_enabled: bool = Field(default=True, alias="VK_ASK_ATTACH_ENABLED")
  vk_ask_attach_max_bytes: int = Field(default=20_971_520, alias="VK_ASK_ATTACH_MAX_BYTES")
  vk_http_max_redirects: int = Field(default=10, alias="VK_HTTP_MAX_REDIRECTS")
  vk_naming_session_ttl_seconds: int = Field(default=3600, alias="VK_NAMING_SESSION_TTL_SECONDS")
  vk_naming_temp_dir: str = Field(
    default_factory=lambda: str(Path(tempfile.gettempdir()) / "vk_bot_uploads"),
    alias="VK_NAMING_TEMP_DIR",
  )
  vk_list_command_prefixes: CsvList = Field(
    default_factory=lambda: ["/list", "/исследования", "/research"],
    alias="VK_LIST_COMMAND_PREFIXES",
  )
  log_level: str = Field(default="INFO", alias="LOG_LEVEL")
  log_format: str = Field(default="text", alias="LOG_FORMAT")

  vk_conversation_history_enabled: bool = Field(
    default=True, alias="VK_CONVERSATION_HISTORY_ENABLED"
  )
  vk_conversation_history_max_turns: int = Field(
    default=5, alias="VK_CONVERSATION_HISTORY_MAX_TURNS"
  )
  vk_conversation_history_ttl_seconds: int = Field(
    default=3600, alias="VK_CONVERSATION_HISTORY_TTL_SECONDS"
  )

  vk_ask_command_prefixes: CsvList = Field(
    default_factory=lambda: ["/ask", "/вопрос", "?", "вопрос:"],
    alias="VK_ASK_COMMAND_PREFIXES",
  )
  vk_idea_command_prefixes: CsvList = Field(
    default_factory=lambda: ["/idea", "/идея"],
    alias="VK_IDEA_COMMAND_PREFIXES",
  )
  vk_greeting_keywords: CsvList = Field(
    default_factory=lambda: ["привет", "старт", "/start", "начать", "hello", "hi"],
    alias="VK_GREETING_KEYWORDS",
  )

  @field_validator(
    "vk_ask_command_prefixes",
    "vk_greeting_keywords",
    "vk_list_command_prefixes",
    "vk_idea_command_prefixes",
    mode="before",
  )
  @classmethod
  def _parse_list_fields(cls, value: object) -> object:
    return _parse_csv_list(value)

  @property
  def effective_redis_url(self) -> str:
    return self.vk_bot_redis_url or self.redis_url

  @property
  def redis_key_prefix(self) -> str:
    return self.vk_bot_redis_key_prefix


def get_settings() -> VkBotSettings:
  return VkBotSettings()
