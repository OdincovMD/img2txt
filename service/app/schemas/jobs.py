"""API schemas for description job status and callback payloads."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


JobStatus = Literal[
    "received",
    "features_ready",
    "classification_ready",
    "generating",
    "completed",
    "error",
]


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus | str


class DescriptionJob(BaseModel):
    job_id: str
    status: JobStatus | str
    important_labels: list[str] | None = None
    classification: dict[str, Any] | None = None
    description: str | None = None
    error: str | None = None
    callback_sent: bool = False
    created_at: str
    updated_at: str


class CallbackPayload(BaseModel):
    status: JobStatus | str
    description: str | None = None
    important_labels: list[str] = Field(default_factory=list)
    error: str | None = None

