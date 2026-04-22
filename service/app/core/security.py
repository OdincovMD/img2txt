"""Service-token authentication helpers."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from service.app.core.config import settings


def require_service_token(
    x_service_token: str | None = Header(default=None, alias="X-Service-Token"),
) -> None:
    if not settings.service_api_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SERVICE_API_TOKEN is not configured",
        )
    if not x_service_token or not secrets.compare_digest(
        x_service_token,
        settings.service_api_token,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service token",
        )
