# API Reference

Документ описывает HTTP-контракт `img2txt Description Service`.

## Общие сведения

- Протокол: HTTP
- Форматы данных: `application/json`, `multipart/form-data`
- Модель взаимодействия: асинхронная обработка задач
- Базовый URL определяется окружением развёртывания

Сервис выполняет три основные операции:

- регистрация задачи по изображению и маске
- приём результата внешней классификации
- чтение текущего состояния задачи

## Аутентификация

Все эндпоинты, кроме `/health`, требуют заголовок:

```http
X-Service-Token: <SERVICE_API_TOKEN>
```

Поведение:

- если `SERVICE_API_TOKEN` не задан в окружении, сервис отвечает `503`
- если токен отсутствует или не совпадает, сервис отвечает `401`

## Модель задачи

Задача идентифицируется внешним `job_id` и хранит:

- текущий статус обработки
- список важных признаков
- payload классификации
- сгенерированное описание
- текст ошибки
- признак успешной доставки callback
- время создания и обновления

Поддерживаемые статусы:

- `received`
- `features_ready`
- `classification_ready`
- `generating`
- `completed`
- `error`

## `GET /health`

Назначение: проверка доступности HTTP-слоя.

Пример ответа:

```json
{
  "status": "ok"
}
```

## `POST /v1/description-jobs`

Назначение: регистрация задачи и запуск этапа извлечения признаков.

### Запрос

Тип: `multipart/form-data`

Поля:

- `job_id`: строковый идентификатор внешней задачи
- `image`: исходное изображение
- `mask`: маска очага

Пример:

```bash
curl -X POST http://127.0.0.1:8000/v1/description-jobs \
  -H "X-Service-Token: <SERVICE_API_TOKEN>" \
  -F "job_id=case-123" \
  -F "image=@./test_images/100.jpg" \
  -F "mask=@./test_images/100_mask.jpg"
```

### Ответ

Успешный ответ:

```json
{
  "job_id": "case-123",
  "status": "received"
}
```

### Поведение

После приёма запроса сервис:

1. создаёт или обновляет состояние задачи
2. ставит фоновую задачу на обработку изображения и маски
3. возвращает ответ до завершения извлечения признаков

### Ошибки

- `422`: пустой `job_id`
- `422`: отсутствует `image` или `mask`
- `401`: невалидный токен
- `503`: сервисный токен не сконфигурирован

## `POST /v1/description-jobs/{job_id}/classification`

Назначение: передача результата внешней классификации.

### Запрос

Тип: `application/json`

Ожидаемое тело:

```json
{
  "feature_type": "Один признак",
  "structure": "Комки",
  "properties": ["Один цвет"],
  "final_class": "Меланома"
}
```

Пример:

```bash
curl -X POST http://127.0.0.1:8000/v1/description-jobs/case-123/classification \
  -H "X-Service-Token: <SERVICE_API_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "feature_type": "Один признак",
    "structure": "Комки",
    "properties": ["Один цвет"],
    "final_class": "Меланома"
  }'
```

### Ответ

Успешный ответ содержит `job_id` и текущий статус после сохранения
классификации:

```json
{
  "job_id": "case-123",
  "status": "classification_ready"
}
```

или

```json
{
  "job_id": "case-123",
  "status": "generating"
}
```

### Поведение

- если признаки ещё не готовы, сервис сохраняет классификацию и оставляет
  задачу в `classification_ready`
- если признаки уже готовы, сервис переводит задачу в `generating` и запускает
  этап генерации описания

### Ошибки

- `422`: пустой `job_id`
- `401`: невалидный токен
- `503`: сервисный токен не сконфигурирован

## `GET /v1/description-jobs/{job_id}`

Назначение: чтение диагностического состояния задачи.

### Ответ

Пример:

```json
{
  "job_id": "case-123",
  "status": "completed",
  "important_labels": ["shape:неправильная"],
  "classification": {
    "feature_type": "Один признак",
    "structure": "Комки",
    "properties": ["Один цвет"],
    "final_class": "Меланома"
  },
  "description": "Клиническое описание...",
  "error": null,
  "callback_sent": true,
  "created_at": "2026-04-23T19:06:34.056876",
  "updated_at": "2026-04-23T19:07:03.978466"
}
```

### Ошибки

- `404`: задача не найдена
- `401`: невалидный токен
- `503`: сервисный токен не сконфигурирован

## Callback Contract

После генерации описания сервис отправляет callback во внешний backend.

### Запрос

```http
POST {CALLBACK_URL}/internal/description-results/{job_id}
X-Service-Token: <CALLBACK_API_TOKEN>
Content-Type: application/json
```

Тело запроса:

```json
{
  "status": "completed",
  "description": "Клиническое описание...",
  "important_labels": ["shape:неправильная"],
  "error": null
}
```

### Поведение

- если `CALLBACK_URL` не задан, callback не выполняется
- если callback успешен, для задачи выставляется `callback_sent=true`
- если callback завершается ошибкой, задача переводится в `error`

## Переменные окружения

Ключевые переменные, влияющие на поведение API:

- `SERVICE_API_TOKEN`: токен авторизации входящих запросов
- `CALLBACK_URL`: базовый URL backend для callback
- `CALLBACK_API_TOKEN`: токен авторизации исходящего callback
- `CALLBACK_TIMEOUT_SECONDS`: таймаут callback-запроса
- `DESCRIPTION_DATABASE_URL`: хранилище состояния задачи
- `DESCRIPTION_TMP_DIR`: временное файловое хранилище

Полный список см. в [service/.env.example](../service/.env.example).
