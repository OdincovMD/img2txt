"""Background job helpers for description generation flow."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from service.app.core import callback, storage
from service.app.core.config import settings
from service.app.pipeline import extract_label_result_from_mask, generate_description_text


def _safe_suffix(filename: str | None, fallback: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix else fallback


def _cleanup(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def maybe_generate_and_callback(job_id: str) -> None:
    job = storage.get_job(job_id)
    if not job:
        return
    if not job.get("important_labels") or not job.get("classification"):
        return
    if job.get("features_only"):
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


def extract_features_task(
    job_id: str,
    image_bytes: bytes,
    mask_bytes: bytes,
    image_name: str | None,
    mask_name: str | None,
    features_only: bool = False,
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
        label_result = extract_label_result_from_mask(image_path, mask_path)
        storage.save_features(
            job_id,
            label_result["important_labels"],
            all_labels=label_result["all_labels"],
            bucketed_labels=label_result["bucketed_labels"],
            features_only=features_only,
        )
        if features_only:
            callback.send_callback(job_id)
        else:
            maybe_generate_and_callback(job_id)
    except Exception as exc:
        storage.save_error(job_id, str(exc))
        try:
            callback.send_callback(job_id)
        except Exception:
            pass
    finally:
        _cleanup(paths)
