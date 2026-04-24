"""Callback delivery to the main skin-cancer-ai backend."""

from __future__ import annotations

from typing import Any

import httpx

from service.app.core.config import settings
from service.app.core import storage


def _payload_from_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": job.get("status"),
        "description": job.get("description"),
        "important_labels": job.get("important_labels") or [],
        "all_labels": job.get("all_labels") or [],
        "bucketed_labels": job.get("bucketed_labels") or [],
        "features_only": bool(job.get("features_only")),
        "error": job.get("error"),
    }


def send_callback(job_id: str) -> None:
    if not settings.callback_url:
        return
    job = storage.get_job(job_id)
    if not job:
        return

    url = f"{settings.callback_url.rstrip('/')}/internal/description-results/{job_id}"
    headers = {}
    if settings.callback_api_token:
        headers["X-Service-Token"] = settings.callback_api_token

    with httpx.Client(timeout=settings.callback_timeout_seconds) as client:
        response = client.post(url, json=_payload_from_job(job), headers=headers)
        response.raise_for_status()
    storage.mark_callback_sent(job_id)
