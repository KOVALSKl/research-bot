"""User-facing UX message templates (RU)."""


def welcome() -> str:
  return (
    "Привет! Я бот для поиска по научным PDF-исследованиям.\n\n"
    "Что я умею:\n"
    "• /ask <вопрос> или /вопрос <вопрос> — поиск по проиндексированным документам\n"
    "• ? <вопрос> — короткая форма команды поиска\n"
    "• PDF во вложении — загрузка и индексация (текст необязателен)\n"
    "• /list — список загруженных исследований\n\n"
    "Напишите «помощь» для списка команд."
  )


def help_text() -> str:
  return (
    "Команды бота:\n"
    "• /ask <вопрос> или /вопрос <вопрос> — поиск по исследованиям\n"
    "• ? <вопрос> — короткая форма поиска\n"
    "• привет, /start — приветствие\n"
    "• /list или /исследования — список загруженных исследований\n"
    "• PDF во вложении — загрузка документа на индексацию\n\n"
    "Поиск запускается только по явной команде."
  )


def unknown_command() -> str:
  return (
    "Я не понял команду. Используйте /ask <вопрос>, ? <вопрос> "
    "или прикрепите PDF-документ."
  )


def unsupported_attachment() -> str:
  return "Поддерживаются только PDF-вложения. Прикрепите PDF или задайте вопрос командой /ask."


def pdf_download_failed() -> str:
  return "Не удалось скачать вложение. Попробуйте отправить PDF ещё раз."


def too_many_attachments(max_count: int) -> str:
  return (
    f"За один раз можно прикрепить не более {max_count} PDF-файлов. "
    f"Отправьте до {max_count} документов и повторите попытку."
  )


def processing_question() -> str:
  return "Идёт обработка вашего вопроса…"


def processing_upload() -> str:
  return "Идёт загрузка и обработка документов…"


def tasks_queued() -> str:
  return (
    "Задачи отправлены в очередь на обработку. "
    "Я сообщу, когда индексация завершится."
  )


def queue_busy() -> str:
  return "Подождите: предыдущая пачка документов ещё обрабатывается."


def rate_limited() -> str:
  return "Слишком много сообщений. Попробуйте через минуту."


def question_too_short(min_length: int) -> str:
  return f"Вопрос слишком короткий. Минимум {min_length} символов."


def no_pdf_or_question() -> str:
  return "Прикрепите PDF-документ или задайте вопрос командой /ask."


def api_error(message: str = "Сервис временно недоступен. Попробуйте позже.") -> str:
  return message


def batch_completed(
  *,
  indexed: int,
  total: int,
  failed_errors: list[str] | None = None,
) -> str:
  text = f"Обработка завершена: {indexed} из {total} документов проиндексировано."
  if failed_errors:
    text += f" Ошибки: {'; '.join(failed_errors)}"
  return text


def ask_display_name(filename: str, index: int, total: int) -> str:
  return (
    f"Введите название для «{filename}» (или «-» для оригинала): {index}/{total}"
  )


def naming_cancelled() -> str:
  return "Ввод названий отменён. Можете снова прикрепить PDF."


def naming_session_busy() -> str:
  return (
    "Сначала завершите ввод названий для текущих файлов или отмените его командой /cancel."
  )


def documents_list_empty() -> str:
  return "Пока нет проиндексированных исследований. Прикрепите PDF."


_STATUS_LABELS = {
  "indexed": "проиндексирован",
  "processing": "обрабатывается",
  "queued": "в очереди",
  "failed": "ошибка",
}


def format_documents_list_header(count: int) -> str:
  return f"Загруженные исследования ({count}):"
