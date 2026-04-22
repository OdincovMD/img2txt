"""Runtime settings for the private description service."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    service_api_token: str = os.environ.get("SERVICE_API_TOKEN", "")
    callback_url: str = os.environ.get("CALLBACK_URL", "")
    callback_api_token: str = os.environ.get("CALLBACK_API_TOKEN", "")
    db_path: str = os.environ.get(
        "DESCRIPTION_DB_PATH",
        "./data/description_service.sqlite3",
    )
    database_url: str = os.environ.get(
        "DESCRIPTION_DATABASE_URL",
        f"sqlite:///{os.environ.get('DESCRIPTION_DB_PATH', './data/description_service.sqlite3')}",
    )
    tmp_dir: str = os.environ.get("DESCRIPTION_TMP_DIR", "/tmp/description-service")
    importance_checkpoint_path: str = os.environ.get(
        "IMPORTANCE_CHECKPOINT_PATH",
        "/app/model/checkpoints/xgb_importance.pkl",
    )
    top_k: int = _get_int("TOP_K", 10)
    description_model: str = os.environ.get(
        "DESCRIPTION_MODEL",
        "openai/gpt-oss-120b",
    )
    callback_timeout_seconds: int = _get_int("CALLBACK_TIMEOUT_SECONDS", 15)


settings = Settings()
