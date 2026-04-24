"""Repository functions for description jobs."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from service.app.db.models import DescriptionJob, utc_now


TERMINAL_STATUSES = {"completed", "error"}


def _serialize(job: DescriptionJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "important_labels": job.important_labels,
        "all_labels": job.all_labels,
        "bucketed_labels": job.bucketed_labels,
        "features_only": job.features_only,
        "classification": job.classification,
        "description": job.description,
        "error": job.error,
        "callback_sent": job.callback_sent,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }


def get_job(session: Session, job_id: str) -> dict[str, Any] | None:
    job = session.get(DescriptionJob, job_id)
    return _serialize(job) if job else None


def upsert_received(session: Session, job_id: str, features_only: bool = False) -> dict[str, Any]:
    job = session.get(DescriptionJob, job_id)
    now = utc_now()
    if job is None:
        job = DescriptionJob(
            job_id=job_id,
            status="received",
            features_only=features_only,
            created_at=now,
            updated_at=now,
        )
        session.add(job)
    elif job.status not in TERMINAL_STATUSES:
        job.status = "received"
        job.features_only = features_only
        job.error = None
        job.updated_at = now
    session.commit()
    session.refresh(job)
    return _serialize(job)


def save_features(
    session: Session,
    job_id: str,
    important_labels: list[str],
    all_labels: list[str] | None = None,
    bucketed_labels: list[str] | None = None,
    features_only: bool = False,
) -> dict[str, Any]:
    job = session.get(DescriptionJob, job_id)
    now = utc_now()
    if job is None:
        job = DescriptionJob(
            job_id=job_id,
            status="features_ready",
            important_labels=important_labels,
            all_labels=all_labels,
            bucketed_labels=bucketed_labels,
            features_only=features_only,
            created_at=now,
            updated_at=now,
        )
        session.add(job)
    else:
        job.important_labels = important_labels
        job.all_labels = all_labels
        job.bucketed_labels = bucketed_labels
        job.features_only = features_only
        job.status = "generating" if job.classification and not job.features_only else "features_ready"
        job.error = None
        job.updated_at = now
    session.commit()
    session.refresh(job)
    return _serialize(job)


def save_classification(session: Session, job_id: str, classification: dict[str, Any]) -> dict[str, Any]:
    job = session.get(DescriptionJob, job_id)
    now = utc_now()
    if job is None:
        job = DescriptionJob(
            job_id=job_id,
            status="classification_ready",
            classification=classification,
            created_at=now,
            updated_at=now,
        )
        session.add(job)
    else:
        job.classification = classification
        if job.important_labels and not job.features_only:
            job.status = "generating"
        elif job.important_labels:
            job.status = "features_ready"
        else:
            job.status = "classification_ready"
        job.error = None
        job.updated_at = now
    session.commit()
    session.refresh(job)
    return _serialize(job)


def save_completed(session: Session, job_id: str, description: str) -> dict[str, Any]:
    job = session.get(DescriptionJob, job_id)
    now = utc_now()
    if job is None:
        job = DescriptionJob(
            job_id=job_id,
            status="completed",
            description=description,
            created_at=now,
            updated_at=now,
        )
        session.add(job)
    else:
        job.status = "completed"
        job.description = description
        job.error = None
        job.updated_at = now
    session.commit()
    session.refresh(job)
    return _serialize(job)


def save_error(session: Session, job_id: str, error: str) -> dict[str, Any]:
    job = session.get(DescriptionJob, job_id)
    now = utc_now()
    if job is None:
        job = DescriptionJob(
            job_id=job_id,
            status="error",
            error=error,
            created_at=now,
            updated_at=now,
        )
        session.add(job)
    else:
        job.status = "error"
        job.error = error
        job.updated_at = now
    session.commit()
    session.refresh(job)
    return _serialize(job)


def mark_callback_sent(session: Session, job_id: str) -> None:
    job = session.get(DescriptionJob, job_id)
    if not job:
        return
    job.callback_sent = True
    job.updated_at = utc_now()
    session.commit()
