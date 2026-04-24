"""Storage facade backed by SQLAlchemy ORM."""

from __future__ import annotations

from typing import Any

from service.app.db import repository
from service.app.db import session as db_session


def init_db() -> None:
    db_session.init_db()


def get_job(job_id: str) -> dict[str, Any] | None:
    with db_session.SessionLocal() as session:
        return repository.get_job(session, job_id)


def upsert_received(job_id: str, features_only: bool = False) -> dict[str, Any]:
    with db_session.SessionLocal() as session:
        return repository.upsert_received(session, job_id, features_only=features_only)


def save_features(
    job_id: str,
    important_labels: list[str],
    all_labels: list[str] | None = None,
    bucketed_labels: list[str] | None = None,
    features_only: bool = False,
) -> dict[str, Any]:
    with db_session.SessionLocal() as session:
        return repository.save_features(
            session,
            job_id,
            important_labels,
            all_labels=all_labels,
            bucketed_labels=bucketed_labels,
            features_only=features_only,
        )


def save_classification(job_id: str, classification: dict[str, Any]) -> dict[str, Any]:
    with db_session.SessionLocal() as session:
        return repository.save_classification(session, job_id, classification)


def save_completed(job_id: str, description: str) -> dict[str, Any]:
    with db_session.SessionLocal() as session:
        return repository.save_completed(session, job_id, description)


def save_error(job_id: str, error: str) -> dict[str, Any]:
    with db_session.SessionLocal() as session:
        return repository.save_error(session, job_id, error)


def mark_callback_sent(job_id: str) -> None:
    with db_session.SessionLocal() as session:
        repository.mark_callback_sent(session, job_id)
