"""HTTP routes for description jobs."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from service.app.core import storage
from service.app.core.description_jobs import extract_features_task, maybe_generate_and_callback
from service.app.core.security import require_service_token
from service.app.schemas.jobs import DescriptionJob, JobStatusResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/v1/description-jobs",
    dependencies=[Depends(require_service_token)],
    response_model=JobStatusResponse,
)
async def create_description_job(
    background_tasks: BackgroundTasks,
    job_id: str = Form(...),
    features_only: bool = Form(False),
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
    storage.upsert_received(clean_job_id, features_only=features_only)
    background_tasks.add_task(
        extract_features_task,
        clean_job_id,
        image_bytes,
        mask_bytes,
        image.filename,
        mask.filename,
        features_only,
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
    if job.get("important_labels") and not job.get("features_only"):
        background_tasks.add_task(maybe_generate_and_callback, clean_job_id)
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
