"""HTTP routes for description jobs."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from service.app.core import callback, storage
from service.app.core.config import settings
from service.app.core.security import require_service_token
from service.app.pipeline import extract_important_labels_from_mask, generate_description_text
from service.app.schemas.jobs import DescriptionJob, JobStatusResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _safe_suffix(filename: str | None, fallback: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix else fallback


def _cleanup(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def _maybe_generate_and_callback(job_id: str) -> None:
    job = storage.get_job(job_id)
    if not job:
        return
    if not job.get("important_labels") or not job.get("classification"):
        return
    if job.get("status") == "completed":
        return

    try:
        description = generate_description_text(
            job["important_labels"],
            job["classification"],
        )
        storage.save_completed(job_id, description)
    except Exception as exc:
        storage.save_error(job_id, str(exc))

    try:
        callback.send_callback(job_id)
    except Exception as exc:
        storage.save_error(job_id, f"Callback failed: {exc}")


def _extract_features_task(
    job_id: str,
    image_bytes: bytes,
    mask_bytes: bytes,
    image_name: str | None,
    mask_name: str | None,
) -> None:
    prefix = f"{job_id}-{uuid4().hex}"
    tmp_dir = Path(settings.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image_path = tmp_dir / f"{prefix}-image{_safe_suffix(image_name, '.jpg')}"
    mask_path = tmp_dir / f"{prefix}-mask{_safe_suffix(mask_name, '.png')}"
    paths = [image_path, mask_path]

    try:
        image_path.write_bytes(image_bytes)
        mask_path.write_bytes(mask_bytes)
        important_labels = extract_important_labels_from_mask(image_path, mask_path)
        storage.save_features(job_id, important_labels)
        _maybe_generate_and_callback(job_id)
    except Exception as exc:
        storage.save_error(job_id, str(exc))
        try:
            callback.send_callback(job_id)
        except Exception:
            pass
    finally:
        _cleanup(paths)


@router.post(
    "/v1/description-jobs",
    dependencies=[Depends(require_service_token)],
    response_model=JobStatusResponse,
)
async def create_description_job(
    background_tasks: BackgroundTasks,
    job_id: str = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
) -> dict[str, str]:
    if not job_id.strip():
        raise HTTPException(status_code=422, detail="job_id is required")

    image_bytes = await image.read()
    mask_bytes = await mask.read()
    if not image_bytes or not mask_bytes:
        raise HTTPException(status_code=422, detail="image and mask are required")

    clean_job_id = job_id.strip()
    storage.upsert_received(clean_job_id)
    background_tasks.add_task(
        _extract_features_task,
        clean_job_id,
        image_bytes,
        mask_bytes,
        image.filename,
        mask.filename,
    )
    return {"job_id": clean_job_id, "status": "received"}


@router.post(
    "/v1/description-jobs/{job_id}/classification",
    dependencies=[Depends(require_service_token)],
    response_model=JobStatusResponse,
)
async def complete_description_job(
    job_id: str,
    payload: dict,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    clean_job_id = job_id.strip()
    if not clean_job_id:
        raise HTTPException(status_code=422, detail="job_id is required")

    job = storage.save_classification(clean_job_id, payload)
    if job.get("important_labels"):
        background_tasks.add_task(_maybe_generate_and_callback, clean_job_id)
    return {"job_id": clean_job_id, "status": str(job["status"])}


@router.get(
    "/v1/description-jobs/{job_id}",
    dependencies=[Depends(require_service_token)],
    response_model=DescriptionJob,
)
def get_description_job(job_id: str) -> dict:
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Description job not found")
    return job
