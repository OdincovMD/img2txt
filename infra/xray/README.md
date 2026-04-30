# Xray Sidecar

Этот вариант нужен, если ваш VPN-клиент экспортирует только `JSON`-конфиг
`Xray/V2Ray`, а не `Clash/Mihomo` YAML.

## Что положить

Экспортируйте профиль из `happ` в:

```text
infra/xray/config.json
```

## Важная правка в экспортированном JSON

Чтобы `description_service` мог достучаться до sidecar по Docker-сети,
inbound-прокси в `config.json` должны слушать не `127.0.0.1`, а `0.0.0.0`.

Проверьте как минимум два inbound-блока:

```json
{
  "listen": "0.0.0.0",
  "port": 10808,
  "protocol": "socks"
}
```

```json
{
  "listen": "0.0.0.0",
  "port": 10809,
  "protocol": "http"
}
```

Если оставить `127.0.0.1`, proxy будет доступен только внутри самого
`xray_proxy` контейнера, а `description_service` до него не дотянется.

## Запуск

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.xray.yml \
  up --build -d
```

## Проверка

Логи `xray`:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.xray.yml \
  logs -f xray_proxy
```

Проверка proxy с хоста:

```bash
curl --proxy http://127.0.0.1:10809 https://api.ipify.org
```

Если внешний IP вернулся, можно считать, что только LLM-вызовы из
`description_service` пойдут через VPN.
