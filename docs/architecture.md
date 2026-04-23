# Architecture Overview

Документ описывает внутреннюю структуру `img2txt Description Service`, модель
обработки задач и основные зависимости.

## Роль сервиса

Сервис является отдельным внутренним компонентом в контуре
[`skin-cancer-ai`](https://github.com/OdincovMD/skin-cancer-ai). Он не строит
маску и не выполняет первичную классификацию.

На вход сервис получает:

- исходное изображение
- маску очага
- результат внешней классификации

На выходе сервис формирует:

- список наиболее значимых признаков
- итоговое клиническое описание
- payload callback-запроса для внешнего backend

## Модель обработки

Обработка построена как асинхронный конвейер задач с хранением состояния.

Состояние задачи хранится в SQLite и проходит через последовательность:

```text
received
  -> features_ready
  -> generating
  -> completed

received
  -> classification_ready
  -> generating
  -> completed

любое активное состояние
  -> error
```

Фактический порядок зависит от того, что приходит первым:

- изображение и маска
- классификация

Генерация описания начинается только после того, как в задаче присутствуют:

- `important_labels`
- `classification`

## Основные слои

### HTTP-слой

Расположение: `service/app/routers`

Отвечает за:

- приём HTTP-запросов
- базовую валидацию входных данных
- проверку `X-Service-Token`
- постановку фоновых задач
- выдачу текущего состояния задачи

Основные маршруты:

- `GET /health`
- `POST /v1/description-jobs`
- `POST /v1/description-jobs/{job_id}/classification`
- `GET /v1/description-jobs/{job_id}`

### Базовая логика

Расположение: `service/app/core`

Отвечает за:

- загрузку конфигурации из переменных окружения
- проверку входного токена
- отправку callback-запросов
- координацию фоновых сценариев обработки задач

Ключевые модули:

- `config.py`
- `security.py`
- `storage.py`
- `callback.py`
- `description_jobs.py`

### Слой хранения

Расположение: `service/app/db`

Отвечает за:

- создание SQLAlchemy engine и session factory
- регистрацию ORM-моделей
- чтение и обновление состояния задач

Хранимые поля:

- `job_id`
- `status`
- `important_labels`
- `classification`
- `description`
- `error`
- `callback_sent`
- `created_at`
- `updated_at`

### Слой обработки

Расположение: `service/app/pipeline.py`

Отвечает за:

- загрузку изображения и маски
- извлечение признаков
- бакетизацию
- ранжирование
- нормализацию payload классификации
- генерацию текста

`pipeline.py` служит координирующим слоем над отдельными вычислительными
этапами.

### Вычислительные этапы

Расположение: `service/app/pipeline_steps`

Содержит:

- `step1_features`: извлечение признаков изображения и маски
- `step2_bucketing`: перевод численных признаков в бакеты
- `step3_ranking`: ранжирование значимых признаков по checkpoint-модели
- `step4_generation`: подготовка prompt и вызов LLM API

## Поток выполнения

### Ветка 1: изображение и маска

1. HTTP-слой принимает `POST /v1/description-jobs`
2. задача создаётся или обновляется со статусом `received`
3. `description_jobs.extract_features_task()` сохраняет временные файлы
4. pipeline извлекает признаки
5. ranking формирует `important_labels`
6. storage сохраняет результат как `features_ready` или `generating`
7. если классификация уже существует, запускается генерация и callback

### Ветка 2: классификация

1. HTTP-слой принимает `POST /v1/description-jobs/{job_id}/classification`
2. storage сохраняет payload классификации
3. статус становится `classification_ready` или `generating`
4. если признаки уже существуют, запускается генерация и callback

### Ветка 3: генерация

1. `maybe_generate_and_callback()` читает состояние задачи
2. проверяет наличие `important_labels` и `classification`
3. вызывает генерацию текста
4. сохраняет итоговое описание как `completed`
5. вызывает callback backend
6. при сбое callback переводит задачу в `error`

## Внешние зависимости

Сервис зависит от следующих компонентов:

- FastAPI и Uvicorn для HTTP-сервера
- SQLite и SQLAlchemy для хранения состояния
- OpenCV, NumPy, scikit-image и связанных библиотек для извлечения признаков
- checkpoint XGBoost-модели для ранжирования
- внешнего LLM API через OpenAI-совместимый клиент
- внешнего backend по `CALLBACK_URL`

## Конфигурация

Ключевые параметры времени выполнения:

- `SERVICE_API_TOKEN`
- `CALLBACK_URL`
- `CALLBACK_API_TOKEN`
- `CALLBACK_TIMEOUT_SECONDS`
- `DESCRIPTION_DATABASE_URL`
- `DESCRIPTION_TMP_DIR`
- `IMPORTANCE_CHECKPOINT_PATH`
- `DESCRIPTION_MODEL`
- `TOP_K`

## Инженерные ограничения

- SQLite подходит для локального хранения состояния, но не для многосервисного
  контура с высокой конкуренцией запросов.
- Временные файлы создаются на локальной файловой системе контейнера.
- Доставка callback зависит от сетевой доступности backend внутри Docker-сети.
- Ошибка callback приводит к конечному состоянию `error`, даже если текст уже
  был успешно сгенерирован.
- Сервис не содержит собственного механизма построения маски.
- Сервис не содержит встроенного классификатора.

## Связанные документы

- [README](../README.md)
- [API Reference](api.md)
- [Deployment and operation](deployment.md)
