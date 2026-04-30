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
- доступ к внешнему LLM API через `LLM_API_KEY`, `OPENAI_API_KEY` или `GROQ_API_KEY`
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
- `LLM_API_KEY`: универсальный ключ для OpenAI-compatible LLM API
- `OPENAI_API_KEY`: fallback-имя переменной для OpenAI-compatible клиентов
- `GROQ_API_KEY`: fallback-имя переменной для Groq
- `LLM_BASE_URL`: OpenAI-compatible base URL для LLM API
- `LLM_PROXY_URL`: optional proxy only for outbound LLM requests from this container
- `IMPORTANCE_CHECKPOINT_PATH`: путь к checkpoint-модели ранжирования
- `DESCRIPTION_DATABASE_URL`: путь к SQLite-хранилищу статусов

## Прокси только для этого контейнера

Если хост-машина не должна поднимать VPN глобально, но внешнее LLM API доступно
только через прокси, используйте отдельную переменную `LLM_PROXY_URL` в
`service/.env`.

Пример:

```env
LLM_API_KEY=...
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_PROXY_URL=http://host.docker.internal:3128
```

В этом режиме через прокси пойдут только LLM-запросы из
`service/app/pipeline_steps/step4_generation/llm_client.py`. Внутренний
`CALLBACK_URL` в backend продолжит вызываться напрямую и не будет зависеть от
маршрута до внешнего API.

Если у вас не HTTP-proxy, а только WireGuard/OpenVPN-конфиг, следующим шагом
имеет смысл поднимать отдельный sidecar-контейнер с VPN и выпускать наружу
только `description_service` через его network namespace.

## Sidecar VPN через Xray JSON

Если ваш клиент VPN экспортирует только `Xray/V2Ray JSON`, используйте overlay:

```text
docker-compose.xray.yml
```

### Подготовка

Экспортируйте рабочий профиль из `happ` в:

```text
infra/xray/config.json
```

Файл должен существовать до запуска `docker compose`. Если его нет, Docker может
создать директорию-заглушку вместо файла, и `xray_proxy` не сможет стартовать.

Проверьте, что inbound для `socks` и `http` слушают `0.0.0.0`, а не
`127.0.0.1`. Иначе `description_service` не сможет достучаться до sidecar по
Docker-сети.

Ожидаемые порты:

- `10808` для `socks`
- `10809` для `http`

### Запуск

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.xray.yml \
  up --build -d
```

### Проверка

Логи:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.xray.yml \
  logs -f xray_proxy
```

Proxy с хоста:

```bash
curl --proxy http://127.0.0.1:10809 https://api.ipify.org
```

Это единственный поддерживаемый sidecar-сценарий в репозитории. Детали и
пример шаблона лежат в [infra/xray/README](../infra/xray/README.md) и
[infra/xray/config.example.json](../infra/xray/config.example.json).

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
