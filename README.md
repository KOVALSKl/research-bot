# research-bot

RAG-ассистент по научным исследованиям. Пользователи задают вопросы по PDF-исследованиям через бота ВКонтакте; поиск и генерация ответов выполняются отдельным API-сервисом.

## Текущий функционал

Research-агент отвечает на вопросы по загруженным PDF и при необходимости дополняет ответ публикациями из OpenAlex, arXiv и Semantic Scholar. Бот ВКонтакте и REST API — клиенты агента.

**Реализовано (v2–v9):** загрузка и индексация PDF с пользовательскими названиями, гибридный поиск, RAG (`POST /ask`), **stateful ReAct research-агент** (`POST /agent/ask`, **`POST /agent/ask/stream`** с SSE-событиями `reasoning` и `progress`), фильтр релевантности, динамический выбор tools (локальный поиск, внешние базы), inline-цитаты `[n]` / `[E n]`, содержательный синтез из контекста, доставка процитированных источников (PDF-вложения и ссылки), команда `/ask` в VK с отображением рассуждений агента (`💭 thought`), команда `/list` для каталога исследований, **оценка идей** (`mode=idea_evaluation`, структурированный `idea_assessment`, команда `/idea` в VK), bootstrap local search и safety net при пустом контексте, контекстные сообщения об ошибках, скачивание PDF-вложений из VK CDN с обработкой HTTP-редиректов, явный mode lock для `/ask` и `/idea`, структурированная оценка релевантности идеи, гарантия выполнения поиска перед синтезом, **доставка PDF внешних публикаций** (open-access, lazy fetch + cache), heuristic rerank внешней литературы.

**Завершено (v12):** корректные HTTP-коды при ошибках storage (503), query reformulation для external search по умолчанию.

**Завершено (v13):** ingest-first upload (staging → fast `202`), archive-after-index на Yandex Disk, `source_url` в метаданных, redirect при скачивании.

**Запланировано (v5, после v6):** бенчмарки и метрики для сравнения конфигураций (embeddings, LLM, режимы поиска), воспроизводимые eval-отчёты.

**Docker Compose (`--profile full`):** требуется Docker Compose v2.1+ (`depends_on` с `condition: service_healthy`). После обновления кода или `docker-compose.yml` (healthcheck qdrant и др.) пересоберите образы без кэша:

```bash
GIT_SHA=$(git rev-parse --short HEAD) docker compose --profile full build vk_bot core_api worker --no-cache
docker compose --profile full up -d
docker logs research-bot-vk_bot-1 2>&1 | grep vk_bot.start
```

**Cold start / readiness:** `qdrant` (healthcheck: bash TCP + `GET /healthz`, без curl в образе) и `redis` должны стать `healthy`, затем `core_api` (healthcheck `GET /health`, `start_period: 60s` при pre-cache моделей в образе, до ~120s без pre-cache). `vk_bot` стартует только после `core_api: healthy` и дополнительно ждёт `/health` при boot. Первый `docker compose --profile full up` может занять до ~2 минут без pre-cache embedding-моделей (~1 ГБ); с pre-cache в образе `core_api`/`worker` — обычно < 1 минуты.

В логах `vk_bot.start` должны быть `git_sha` и `redirect_handler=shared`. Без пересборки образ может содержать старый код (например, ошибка HTTP 302 при скачивании PDF).

**Post-deploy проверка PDF upload (v13):**
1. Отправьте PDF-файл боту в VK (без текста или с любым сообщением).
2. В логах `vk_bot` должно появиться `attachment.download_success` (не `HTTPStatusError: 302 Found`).
3. Бот должен перейти к wizard именования и подтвердить постановку в очередь (`upload.queued`) — **без ошибки ReadTimeout**.
4. `POST /documents` возвращает `202` за секунды; Yandex Disk API **не** вызывается на upload route.
5. Worker индексирует PDF в Qdrant, затем архивирует на Yandex Disk и записывает `source_url`.
6. Документ появляется в `/list` со статусом `indexed`; `GET /documents/files/{research_id}` отдаёт redirect на `source_url` (или stream из archive).
7. `/ask` находит контент по загруженному PDF.

### Локальный корпус и бот

- Монорепозиторий с сервисами **core_api** (FastAPI), **worker** (Celery), **vk_bot** (бот ВКонтакте) и общей библиотекой **research_shared**.
- Docker Compose с Qdrant (векторное хранилище) и Redis (очередь фоновых задач, кэш внешней литературы и состояние бота).
- **Бот ВКонтакте**: приём сообщений через Long Polling или Callback API. PDF-вложения скачиваются через VK CDN с обработкой HTTP-редиректов (301/302/…). Поиск — по командам `/ask`, `/idea`, `/вопрос`, `?`; PDF без текста — на индексацию.
- **Загрузка PDF с пользовательскими названиями**: после прикрепления файлов бот поочерёдно спрашивает отображаемое имя (или `-` для оригинального). Эти имена используются в цитатах и в списке исследований.
- **Команда `/list`** (и `/исследования`, `/research`) — просмотр каталога загруженных исследований со статусом индексации.
- **Загрузка PDF через API или бота**: парсинг, чанкинг, эмбеддинги и индексация в фоне. После индексации исходник архивируется в облачном хранилище (v10/v13: Yandex Disk `research-docs/`; dev: локально в `researches/`).
- Гибридный поиск (семантический + по терминам) с фильтрацией по документу, авторам, страницам и главе.
- **Вопрос-ответ** (`POST /agent/ask`, команда `/ask` в VK): развёрнутый ответ с inline-цитатами `[1]`, `[2]`, `[E1]` и блоками источников — локальные PDF (с вложением файла в VK) и внешние публикации (со ссылками). Legacy `POST /ask` — упрощённый RAG без agent pipeline.
- Защита бота: фильтрация исходящих, dedup сообщений, rate limit, санитизация ввода.
- **Структурированные логи** (JSON) в vk_bot и core_api.

### Внешняя литература (фундамент v11)

- Поиск публикаций в **OpenAlex**, **arXiv** и опционально **Semantic Scholar**.
- Единый формат результата: название, авторы, год, abstract, DOI, URL, источник.
- Кэширование ответов в Redis.
- Отладочный эндпоинт **`POST /literature/search`**.

### Research Agent

- **ReAct-цикл (v6):** агент сам решает, когда искать локально, во внешних базах и когда завершать сбор контекста; рассуждения (`thought`) видны пользователю в VK/SSE.
- **Production fixes (v7):** bootstrap local search перед ReAct, safety net при пустом контексте, progress+reasoning в VK status.
- **Production quality (v8):** mode lock для `/ask`/`/idea`, structured relevance, search attestation, deploy verification.
- **Production fixes (v9):** external PDF delivery, heuristic rerank, pdf_url в провайдерах.
- **Cloud storage & quality (v10, в работе):** Yandex Disk, VK deploy hardening, external search и idea evaluation.
- **`POST /agent/ask`**: ответ с локальными и внешними источниками, лог шагов `steps[]` с `thought`.
- **`POST /agent/ask/stream`**: SSE-события `reasoning`, `progress` (classify/synthesize) и финальный `complete`.
- Фильтр релевантности после локального поиска; post-validation shallow answers и evidence.
- Цитирование: локальные `[n]`, внешние `[E n]`; в VK — группировка по документу (стр. 5, 12, 23), PDF-вложения для локальных и внешних open-access источников.
- VK: команда `/ask` через streaming API; `/idea` — структурированная оценка идеи; рассуждения агента (`💭`), preview внешних публикаций, финальный ответ с источниками.
- Настройки: `AGENT_MAX_ITERATIONS` (лимит итераций ReAct, default 6).

## Roadmap (запланировано)

- **v13** — ingest-first upload, archive-after-index, source URL в метаданных
- **v5** — бенчмарки и метрики для сравнения конфигураций (embeddings, LLM, режимы поиска)

## Быстрый старт

```bash
cp .env.example .env
# При первом запуске fastembed скачает multilingual-e5-large (~1 ГБ)
# Ollama нужна только для LLM-ответов (если LLM_ENABLED=true)

docker compose up -d qdrant redis
uv sync
uv run core-api
# фоновая обработка загруженных документов (в отдельном терминале):
uv run celery -A worker.celery_app worker
# бот ВКонтакте (нужны VK_BOT_TOKEN и VK_GROUP_ID в .env):
uv run vk-bot
```

API доступен на `http://localhost:8000`, документация — `/docs`.

### Эндпоинты core_api

| Метод | Путь | Назначение |
|-------|------|------------|
| `GET` | `/health` | Проверка состояния |
| `POST` | `/documents` | Загрузка одного PDF (multipart). Опционально `display_name`. Принимает файл во временное staging-хранилище и ставит фоновую задачу → `202 {task_id, research_id}`; архивация на Yandex Disk выполняется worker'ом после индексации. При `INGEST_SYNC=true` (dev) обрабатывает inline → `201` |
| `POST` | `/documents/batch` | Мультизагрузка PDF (опционально `display_names` для каждого файла) → `202 {jobs, errors}` |
| `GET` | `/documents` | Список загруженных исследований (`research_id`, `display_name`, статус, число чанков) |
| `GET` | `/documents/tasks/{task_id}` | Статус фоновой обработки |
| `POST` | `/documents/chunks` | Прямой upsert готовых `ResearchChunk` (legacy) |
| `DELETE` | `/documents/{chunk_id}` · `/documents/research/{research_id}` | Удаление по chunk id / по документу |
| `POST` | `/search` | Гибридный поиск → `list[SearchResult]`; опционально `filters` |
| `POST` | `/ask` | Вопрос → контекст с цитатами; `answer` генерируется LLM при `LLM_ENABLED=true`, иначе `null` |
| `POST` | `/literature/search` | Поиск по открытым научным базам → `list[ExternalPaper]` |
| `POST` | `/agent/ask` | Research Agent: local → relevance filter → external fallback → ответ с `steps[]` |
| `POST` | `/agent/ask/stream` | То же через SSE: события `progress` + `complete` |

Порядок запуска: `qdrant` + `redis` (Docker) → `core-api` → `worker` (Celery) → `vk-bot` (опционально). LLM-генерация и фоновый сканер по умолчанию выключены.

**Ingest policy (v13):** production использует `INGEST_SYNC=false` (default). Upload route пишет только в локальный staging (`INGEST_STAGING_DIR`, shared volume `ingest_staging` в Docker для `core_api` + `worker`), возвращает `202` и ставит Celery-задачу. Worker читает staging → индексирует в Qdrant → архивирует на Yandex Disk → удаляет staging. `INGEST_SYNC=true` — только для локальной отладки без worker.

### Бот ВКонтакте

Команды:
- `/ask <вопрос>` или `/вопрос <вопрос>` — поиск по исследованиям (минимум 12 символов)
- `/list`, `/исследования`, `/research` — список загруженных исследований
- `привет`, `/start`, `помощь` — приветствие и справка
- PDF во вложении (с текстом или без) — загрузка на индексацию; бот попросит ввести название для каждого файла (или `-` для оригинального имени)

Произвольный текст без команды не запускает поиск — бот подскажет доступные команды.

```bash
VK_BOT_TOKEN=...
VK_GROUP_ID=...
VK_TRANSPORT=long_polling   # или callback для prod с публичным URL
CORE_API_BASE_URL=http://localhost:8000
```

При `VK_TRANSPORT=callback` дополнительно: `VK_CALLBACK_SECRET`, `VK_CALLBACK_CONFIRMATION`, настройка URL в сообществе VK.

### LLM и RAG

Генерация ответа **выключена по умолчанию** (`LLM_ENABLED=false`). Без LLM эндпоинт `/ask` возвращает найденные фрагменты, цитаты и `answer=null`.

```bash
ASK_DEFAULT_LIMIT=10          # число чанков в контексте RAG
VK_ASK_DEFAULT_LIMIT=10       # то же для бота ВКонтакте
# RAG_SYSTEM_PROMPT=...       # опционально: свой system prompt
```

При `LLM_ENABLED=true` по умолчанию используется **Hugging Face Inference API**:

```bash
LLM_ENABLED=true
LLM_PROVIDER=huggingface
HF_API_TOKEN=hf_...
HF_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
```

Для локальной разработки без HF-токена — Ollama:

```bash
LLM_ENABLED=true
LLM_PROVIDER=ollama
OLLAMA_CHAT_MODEL=qwen3:0.6b
```

### Фоновый сканер `researches/`

Периодическая задача `scan_researches` (Celery beat) индексирует новые/изменённые PDF в `researches/`. **По умолчанию выключена** (`RESEARCHES_SCAN_ENABLED=false`).

### Миграция на v2.3+ (переиндексация)

После обновления до v2.3 схема payload в Qdrant изменилась. Старые точки не содержат полей `source_path`, `authors`, `page`, `chapter` — нужна переиндексация через повторную загрузку PDF или фоновый сканер.

> При смене dense-модели коллекция Qdrant пересоздаётся автоматически (dim 384→1024). Нужна переиндексация документов.

## Данные

PDF-исследования хранятся в Yandex Disk (`research-docs/`, v10) или локально в `researches/` при `STORAGE_BACKEND=local`. Кэш внешних PDF — `research-docs/external/`.
