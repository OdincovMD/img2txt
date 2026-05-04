"""Single-image description pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from service.app.pipeline_steps.step1_features.derived_features import merge_with_composite_features
from service.app.pipeline_steps.step1_features.features import (
    extract_border_features,
    extract_global_color_features_with_mask,
    extract_local_color_features_with_mask,
    extract_shape_features,
    extract_texture_features,
)
from service.app.pipeline_steps.step2_bucketing.bucketing import bucket_columns_from_labels, row_to_labels
from service.app.pipeline_steps.step3_ranking.ranking import rank_important_labels
from service.app.pipeline_steps.step4_generation.description import generate_description_from_labels
from service.app.core.config import settings
from service.app.schemas.classification import normalize_classification_payload


def _read_mask(mask_path: str | Path, expected_shape: tuple[int, int]) -> Any:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot load mask: {mask_path}")
    if mask.shape[:2] != expected_shape:
        mask = cv2.resize(
            mask,
            (expected_shape[1], expected_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    return binary


def _label_entries(labels: dict[str, str]) -> list[str]:
    return [f"{key}:{value}" for key, value in labels.items()]


def extract_label_result_from_mask(
    image_path: str | Path,
    mask_path: str | Path,
) -> dict[str, list[str]]:
    image = _load_image_bgr(image_path)
    mask = _read_mask(mask_path, image.shape[:2])
    features = extract_features(image, mask)
    labels = row_to_labels(features)
    buckets = bucket_columns_from_labels(labels)
    important_labels = rank_important_labels(
        {**features, **buckets},
        importance_model_path=settings.importance_checkpoint_path,
        k=settings.top_k,
    )
    return {
        "important_labels": important_labels,
        "all_labels": _label_entries(labels),
        "bucketed_labels": _label_entries(buckets),
    }


def extract_important_labels_from_mask(
    image_path: str | Path,
    mask_path: str | Path,
) -> list[str]:
    return extract_label_result_from_mask(image_path, mask_path)["important_labels"]


def _load_image_bgr(image_path: str | Path) -> Any:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return image


def extract_features(image: Any, mask: Any) -> dict[str, Any]:
    features: dict[str, Any] = {}
    features.update(extract_global_color_features_with_mask(image, mask))
    features.update(extract_local_color_features_with_mask(image, mask))
    features.update(extract_shape_features(mask))
    features.update(extract_border_features(mask))
    features.update(extract_texture_features(image, mask))
    return merge_with_composite_features(features)


def normalize_classification(payload: dict[str, Any]) -> dict[str, Any]:
    return normalize_classification_payload(payload)


def generate_description_text(
    important_labels: list[str],
    classification_payload: dict[str, Any],
) -> str:
    classification = normalize_classification(classification_payload)
    return generate_description_from_labels(
        important_labels,
        classification,
        model_name=settings.description_model,
        max_tokens=2048,
    )
