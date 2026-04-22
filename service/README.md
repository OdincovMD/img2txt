# Private Description Service

Закрытый HTTP-сервис для генерации клинического описания дерматоскопического
изображения. Основной сценарий: `skin-cancer-ai` строит маску, отправляет сюда
изображение и PNG-маску, затем отдельным запросом передаёт результат
классификации.

## API

Все рабочие эндпоинты, кроме `/health`, требуют заголовок:

```http
X-Service-Token: <SERVICE_API_TOKEN>
```

### `GET /health`

Проверка доступности сервиса.

### `POST /v1/description-jobs`

`multipart/form-data`:

- `job_id`: идентификатор classification job из основного приложения;
- `image`: исходное изображение;
- `mask`: PNG-маска очага.

Сервис сохраняет только признаки и статус. Временные файлы изображения и маски
удаляются после обработки.

### `POST /v1/description-jobs/{job_id}/classification`

JSON результата классификации:

```json
{
  "feature_type": "Один признак",
  "structure": "Комки",
  "properties": ["Один цвет"],
  "final_class": "Меланома"
}
```

После готовности признаков и классификации сервис генерирует описание и отправляет
callback в основной backend:

```http
POST {CALLBACK_URL}/internal/description-results/{job_id}
X-Service-Token: <CALLBACK_API_TOKEN>
```

Тело callback:

```json
{
  "status": "completed",
  "description": "...",
  "important_labels": ["shape:округлая"],
  "error": null
}
```

### `GET /v1/description-jobs/{job_id}`

Диагностический статус задачи.

## Docker

```bash
cp service/.env.example service/.env
docker compose up --build
```

В production не публикуйте порт сервиса наружу. Подключайте его к внутренней
Docker-сети основного приложения или запускайте на host-only интерфейсе.

## Структура

```text
service/
├── Dockerfile
├── requirements.txt
└── app/
    ├── main.py
    ├── routers/          # HTTP API
    ├── schemas/          # API/domain schemas
    ├── core/             # config, auth, SQLite, callback
    ├── pipeline.py       # orchestration for one image
    └── pipeline_steps/   # feature extraction, bucketing, ranking, generation
```
