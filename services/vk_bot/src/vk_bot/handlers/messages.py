"""User-facing UX message templates (RU)."""


def welcome() -> str:
  return (
    "Привет! Я бот для поиска по научным исследованиям.\n\n"
    "Сначала ищу ответ в ваших загруженных PDF, а если локального "
    "контекста недостаточно — дополнительно обращаюсь к открытым базам "
    "(OpenAlex, arXiv и др.) и формирую ответ с цитатами.\n\n"
    "Кратко:\n"
    "• /ask <вопрос> — задать вопрос по исследованиям\n"
    "• /idea <описание> — оценить научную идею\n"
    "• PDF во вложении — загрузить документ в ваш корпус\n"
    "• /list — посмотреть загруженные исследования\n\n"
    "Напишите «помощь» — там подробнее про каждую команду."
  )


def help_text() -> str:
  return (
    "Команды бота и как они работают:\n\n"
    "▸ /ask <вопрос> или /вопрос <вопрос>\n"
    "  Задаёт вопрос research-агенту. Сначала выполняется поиск по вашим "
    "проиндексированным PDF (гибридный: по смыслу и по ключевым словам). "
    "Если найденного контекста мало, агент дополнительно ищет публикации "
    "в открытых научных базах (OpenAlex, arXiv; при наличии ключа — "
    "Semantic Scholar).\n"
    "  Ответ содержит текст с inline-цитатами [1], [2]… для локальных "
    "источников и [E1], [E2]… для внешних публикаций, а также списки "
    "«Локальные источники» и «Внешние публикации».\n"
    "  Пример: /ask Какие модели описывают финансовые пирамиды?\n\n"
    "▸ /idea <описание> или /идея <описание>\n"
    "  Оценивает научную идею или гипотезу: релевантность существующим работам, "
    "аргументы за и против, перспективы успеха — на основе ваших PDF и открытых "
    "публикаций. Ответ структурирован: релевантность, evidence, outlook, уверенность.\n"
    "  Пример: /idea Насколько перспективно применять GNN к выявлению мошенничества?\n\n"
    "▸ ? <вопрос>\n"
    "  Короткая форма команды /ask — работает так же.\n\n"
    "▸ PDF во вложении (без текста или с подписью)\n"
    "  Загружает один или несколько PDF в ваш корпус. После загрузки бот "
    "поочерёдно спросит отображаемое название каждого файла (или «-», "
    "чтобы оставить оригинальное имя). Документы ставятся в очередь на "
    "разбор, чанкинг и индексацию; по готовности их можно искать через /ask.\n"
    "  Во время ввода названий: /cancel или «отмена» — прервать загрузку.\n\n"
    "▸ /list, /исследования или /research\n"
    "  Показывает каталог ваших загруженных исследований: название, статус "
    "индексации и число фрагментов в базе.\n\n"
    "▸ привет, /start, «начать»\n"
    "  Приветствие и краткое описание возможностей бота.\n\n"
    "▸ помощь, /help, «команды»\n"
    "  Эта справка.\n\n"
    "Важно:\n"
    "• Поиск и генерация ответа запускаются только по явной команде "
    "(/ask, /вопрос, /idea, ?) — произвольный текст без команды не обрабатывается "
    "как вопрос.\n"
    "• Поддерживаются только PDF-вложения.\n"
    "• Минимальная длина вопроса — 12 символов."
  )


def unknown_command() -> str:
  return (
    "Я не понял команду. Используйте /ask <вопрос>, /idea <описание идеи>, "
    "? <вопрос> или прикрепите PDF-документ."
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


def agent_reasoning(
  thought: str,
  action_summary: str | None = None,
  *,
  action: str | None = None,
) -> str:
  body = thought.strip()
  if not body and action_summary:
    body = action_summary.strip()
  if not body and action:
    action_labels = {
      "local_hybrid_search": "Ищу в загруженных PDF…",
      "external_literature_search": "Ищу во внешних научных базах…",
      "reformulate_queries": "Уточняю поисковые запросы…",
      "finish": "Формирую ответ…",
    }
    body = action_labels.get(action, "Обрабатываю запрос…")
  if not body:
    body = "Обрабатываю запрос…"
  text = f"💭 {body}"
  if action_summary and action_summary.strip() and action_summary.strip() != body:
    text = f"{text}\n→ {action_summary.strip()}"
  return text


def external_search_started() -> str:
  return "Ищу публикации в открытых научных базах…"


def external_search_results(papers: list) -> str:
  if not papers:
    return external_search_started()
  lines = ["Найдено публикаций:"]
  for paper in papers[:5]:
    title = getattr(paper, "title", paper.get("title", ""))
    url = getattr(paper, "url", paper.get("url", ""))
    lines.append(f"• {title} — {url}")
  return "\n".join(lines)


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


def service_starting() -> str:
  return "Сервис ответов ещё запускается. Попробуйте через минуту."


def service_unavailable() -> str:
  return "Сервис ответов временно недоступен. Попробуйте позже."


def connection_error() -> str:
  return "Сервис ответов временно недоступен. Попробуйте через минуту."


def ask_timeout() -> str:
  return (
    "Генерация ответа заняла слишком много времени. "
    "Попробуйте короче сформулировать вопрос или увеличьте OLLAMA_TIMEOUT_SECONDS."
  )


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
