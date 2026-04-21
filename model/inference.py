"""Inference for the XGBoost importance model."""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from bucketing.schema import bucket_column_name
from model.config import LABEL_NAMES
from model.model import (
    apply_calibration_bias,
    build_features_for_inference,
    get_xgb_model_type,
    load_checkpoint,
    predict_xgboost_scores,
)

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def _row_labels_from_buckets(row: pd.Series) -> dict[str, str]:
    labels = {}
    for key in LABEL_NAMES:
        value = row.get(bucket_column_name(key))
        if isinstance(value, str):
            if ":" in value:
                value = value.split(":", 1)[1]
            labels[key] = value
    return labels


def _top_indices_to_labels(top_indices: List[int], labels_dict: dict[str, str], k: int = 10) -> List[str]:
    result = []
    for idx in top_indices:
        if idx >= len(LABEL_NAMES):
            continue
        key = LABEL_NAMES[idx]
        value = labels_dict.get(key)
        if isinstance(value, str):
            result.append(f"{key}:{value}")
        if len(result) >= k:
            break
    return result[:k]


def _predict_row(row: pd.Series, checkpoint: dict, k: int = 10) -> List[str]:
    labels_dict = _row_labels_from_buckets(row)
    X = build_features_for_inference(pd.DataFrame([row]), checkpoint)
    full_preds = predict_xgboost_scores(
        checkpoint["models"],
        X,
        checkpoint["valid_target_indices"],
        model_type=get_xgb_model_type(checkpoint),
        chain_order=checkpoint.get("chain_order"),
    )[0]
    full_preds = apply_calibration_bias(full_preds.reshape(1, -1), checkpoint.get("calibration_bias"))[0]

    top_idx = np.argsort(full_preds)[::-1][:k].tolist()
    return _top_indices_to_labels(top_idx, labels_dict, k=k)


def rank_features_batch(
    df: pd.DataFrame,
    importance_model_path: Optional[Union[str, Path]] = None,
    k: int = 10,
    verbose: bool = True,
    **_kwargs,
) -> pd.DataFrame:
    result_df = df.copy()

    if importance_model_path is None or xgb is None:
        result_df["important_labels"] = [[] for _ in range(len(result_df))]
        return result_df

    try:
        checkpoint = load_checkpoint(importance_model_path)
        X = build_features_for_inference(result_df, checkpoint)
        full_preds = predict_xgboost_scores(
            checkpoint["models"],
            X,
            checkpoint["valid_target_indices"],
            model_type=get_xgb_model_type(checkpoint),
            chain_order=checkpoint.get("chain_order"),
        )
        full_preds = apply_calibration_bias(full_preds, checkpoint.get("calibration_bias"))
        iterator = tqdm(result_df.iterrows(), total=len(result_df)) if verbose else result_df.iterrows()
        labels = []
        for row_idx, (_, row) in enumerate(iterator):
            labels_dict = _row_labels_from_buckets(row)
            top_idx = np.argsort(full_preds[row_idx])[::-1][:k].tolist()
            labels.append(_top_indices_to_labels(top_idx, labels_dict, k=k))
        result_df["important_labels"] = labels
    except Exception as exc:
        print(f"Warning: Could not rank features: {exc}")
        result_df["important_labels"] = [[] for _ in range(len(result_df))]

    return result_df
