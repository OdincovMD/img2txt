# Deployment and Operation

Этот документ описывает запуск и минимальные эксплуатационные требования для
`img2txt Description Service`.

## Назначение

Сервис предназначен для внутреннего запуска в контуре
[`skin-cancer-ai`](https://github.com/OdincovMD/skin-cancer-ai) и не
рассматривается как публичный internet-facing API.

## Требования

- Docker и Docker Compose
- доступный checkpoint ранжирующей модели в `./model/checkpoints`
- доступ к внешнему LLM API через `GROQ_API_KEY`
- доступность backend по `CALLBACK_URL` из Docker-сети сервиса

## Конфигурация

Создайте локальный файл окружения:

```bash
cp service/.env.example service/.env
```

Критичные переменные:

- `SERVICE_API_TOKEN`: токен для входящих запросов к сервису
- `CALLBACK_URL`: адрес backend, принимающего callback
- `CALLBACK_API_TOKEN`: токен для исходящих callback-запросов
- `GROQ_API_KEY`: ключ для вызова LLM API
- `IMPORTANCE_CHECKPOINT_PATH`: путь к checkpoint-модели ранжирования
- `DESCRIPTION_DATABASE_URL`: путь к SQLite-хранилищу статусов

## Запуск

```bash
docker compose up --build
```

Compose поднимает один сервис `description_service` и подключает:

- `./model/checkpoints` в `/app/model/checkpoints` только на чтение
- именованный volume для `/data`

Порт наружу не публикуется. В `docker-compose.yml` используется `expose: 8000`,
поэтому сервис доступен только другим контейнерам в той же Docker-сети.

## Проверка работоспособности

Проверка health endpoint из контейнера:

```bash
docker compose exec description_service curl -s http://127.0.0.1:8000/health
```

Ожидаемый ответ:

```json
{"status":"ok"}
```

## Эксплуатационные замечания

- Если `SERVICE_API_TOKEN` не задан, рабочие эндпоинты будут отвечать `503`.
- Если `CALLBACK_URL` не резолвится или недоступен, задача может завершиться со
  статусом `error` после успешной генерации описания.
- SQLite используется как внутреннее хранилище состояния job и не предназначен
  для многосервисной конкурентной эксплуатации.
- Временные файлы изображения и маски создаются в `DESCRIPTION_TMP_DIR` и
  удаляются после обработки.

## Связанные документы

- [README](../README.md)
- [API Reference](api.md)
- [Architecture Overview](architecture.md)
