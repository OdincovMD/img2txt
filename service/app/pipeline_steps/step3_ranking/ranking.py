"""XGBoost-based important-feature ranking for one image."""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from service.app.pipeline_steps.step1_features.derived_features import COMPOSITE_FEATURE_COLUMNS
from service.app.pipeline_steps.step1_features.feature_schema import FEATURE_COLUMNS
from service.app.pipeline_steps.step2_bucketing.bucket_schema import bucket_column_name


LABEL_NAMES: list[str] = [
    "asymmetry",
    "borders",
    "contrast",
    "palette",
    "elongation",
    "rim",
    "color_homogeneity",
    "shape",
    "dominant_hue",
    "structure_order",
    "perimeter",
    "fractal_dimension",
    "eccentricity",
    "color_distance_euclidean",
    "delta_H_center_periphery",
    "delta_S_center_periphery",
    "delta_V_center_periphery",
    "delta_V_inner_rim",
    "delta_V_left_right",
    "delta_V_top_bottom",
    "delta_S_left_right",
    "std_H_lesion",
]
NUM_LABELS = len(LABEL_NAMES)
DEFAULT_FEATURE_SET = "selected_buckets"
DEFAULT_XGB_MODEL_TYPE = "xgb"


@lru_cache(maxsize=2)
def load_checkpoint(path: str) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        checkpoint = pickle.load(handle)

    models = []
    for raw in checkpoint["models_raw"]:
        model = xgb.Booster()
        model.load_model(bytearray(raw))
        models.append(model)

    checkpoint["models"] = models
    checkpoint.setdefault("model_type", DEFAULT_XGB_MODEL_TYPE)
    checkpoint.setdefault("feature_set", DEFAULT_FEATURE_SET)
    checkpoint.setdefault("chain_order", list(checkpoint["valid_target_indices"]))
    checkpoint.setdefault("calibration_bias", None)
    checkpoint.pop("models_raw", None)
    return checkpoint


def _bucket_feature_columns(row: dict[str, Any], feature_set: str) -> list[str]:
    if feature_set == "numeric_only":
        return []
    all_bucket_cols = sorted(key for key in row if key.startswith(bucket_column_name("")))
    if feature_set == "all_buckets":
        return all_bucket_cols
    ordered = [bucket_column_name(label) for label in LABEL_NAMES]
    return [col for col in ordered if col in row]


def _model_matrix(row: dict[str, Any], checkpoint: dict[str, Any]) -> pd.DataFrame:
    feature_set = checkpoint.get("feature_set", DEFAULT_FEATURE_SET)
    numeric_cols = [col for col in (*FEATURE_COLUMNS, *COMPOSITE_FEATURE_COLUMNS) if isinstance(row.get(col), (int, float))]
    bucket_cols = _bucket_feature_columns(row, feature_set)

    frames = []
    if numeric_cols:
        frames.append(pd.DataFrame([{col: float(row.get(col) or 0.0) for col in numeric_cols}], dtype=np.float32))
    if bucket_cols:
        bucket_df = pd.DataFrame([{col: str(row.get(col) or "Unknown") for col in bucket_cols}])
        frames.append(pd.get_dummies(bucket_df, columns=bucket_cols, dtype=np.float32))

    X = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=[0])
    missing_cols = [col for col in checkpoint["feature_columns"] if col not in X.columns]
    if missing_cols:
        X = pd.concat([X, pd.DataFrame(0.0, index=X.index, columns=missing_cols, dtype=np.float32)], axis=1)
    return X[checkpoint["feature_columns"]]


def _scores_to_logits(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(scores, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _apply_calibration_bias(scores: np.ndarray, bias: list[float] | np.ndarray | None) -> np.ndarray:
    if bias is None:
        return scores
    bias_arr = np.asarray(bias, dtype=np.float32)
    return _scores_to_logits(scores) + bias_arr


def _append_chain_features(X: pd.DataFrame, target_indices: list[int], predictions: list[np.ndarray]) -> pd.DataFrame:
    frame = X.copy()
    for target_idx, preds in zip(target_indices, predictions):
        frame[f"chain_pred__{LABEL_NAMES[target_idx]}"] = np.asarray(preds, dtype=np.float32)
    return frame


def _predict_scores(checkpoint: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    models = checkpoint["models"]
    valid_target_indices = checkpoint["valid_target_indices"]
    full_preds = np.zeros((len(X), NUM_LABELS), dtype=np.float32)

    if checkpoint.get("model_type") == "xgb_classifier_chain":
        active_chain = list(checkpoint.get("chain_order") or valid_target_indices)
        previous_preds: list[np.ndarray] = []
        for model_idx, target_idx in enumerate(active_chain):
            chain_features = _append_chain_features(X, active_chain[:model_idx], previous_preds)
            preds = models[model_idx].predict(xgb.DMatrix(chain_features), output_margin=False)
            full_preds[:, target_idx] = preds
            previous_preds.append(preds.astype(np.float32))
        return full_preds

    dmatrix = xgb.DMatrix(X)
    for model_idx, target_idx in enumerate(valid_target_indices):
        full_preds[:, target_idx] = models[model_idx].predict(dmatrix, output_margin=False)
    return full_preds


def _row_labels_from_buckets(row: dict[str, Any]) -> dict[str, str]:
    labels = {}
    for key in LABEL_NAMES:
        value = row.get(bucket_column_name(key))
        if isinstance(value, str):
            labels[key] = value.split(":", 1)[1] if ":" in value else value
    return labels


def rank_important_labels(
    row: dict[str, Any],
    importance_model_path: str,
    k: int = 10,
) -> list[str]:
    checkpoint = load_checkpoint(importance_model_path)
    X = _model_matrix(row, checkpoint)
    scores = _predict_scores(checkpoint, X)
    scores = _apply_calibration_bias(scores, checkpoint.get("calibration_bias"))
    top_indices = np.argsort(scores[0])[::-1][:k].tolist()
    labels_dict = _row_labels_from_buckets(row)

    labels = []
    for idx in top_indices:
        if idx >= len(LABEL_NAMES):
            continue
        key = LABEL_NAMES[idx]
        value = labels_dict.get(key)
        if isinstance(value, str):
            labels.append(f"{key}:{value}")
        if len(labels) >= k:
            break
    return labels
