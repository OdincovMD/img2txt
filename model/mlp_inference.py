"""Inference for the MLP importance model."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from bucketing.schema import bucket_column_name
from model.config import LABEL_NAMES, get_torch_device
from model.mlp import build_mlp_feature_matrix, load_mlp_checkpoint, transform_features


def _row_labels_from_buckets(row: pd.Series) -> dict[str, str]:
    labels = {}
    for key in LABEL_NAMES:
        value = row.get(bucket_column_name(key))
        if isinstance(value, str):
            labels[key] = value
    return labels


def _top_indices_to_labels(top_indices: list[int], labels_dict: dict[str, str], k: int = 10) -> list[str]:
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


def build_mlp_features_for_inference(
    feat_df: pd.DataFrame,
    checkpoint: dict[str, object],
) -> np.ndarray:
    feature_set = checkpoint.get("feature_set", "numeric_only")
    if not isinstance(feature_set, str):
        feature_set = "numeric_only"
    X, _ = build_mlp_feature_matrix(feat_df, feature_set=feature_set)  # type: ignore[arg-type]
    return transform_features(
        X,
        checkpoint.get("scaler"),
        checkpoint.get("numeric_feature_columns", []),
        feature_columns=checkpoint["feature_columns"],
    )


def rank_features_batch_mlp(
    df: pd.DataFrame,
    importance_model_path: Optional[Union[str, Path]] = None,
    k: int = 10,
    verbose: bool = True,
    **_kwargs,
) -> pd.DataFrame:
    result_df = df.copy()
    if importance_model_path is None:
        result_df["important_labels"] = [[] for _ in range(len(result_df))]
        return result_df

    try:
        checkpoint = load_mlp_checkpoint(importance_model_path)
        device = get_torch_device()
        model = checkpoint["model"].to(device)
        model.eval()
        X = build_mlp_features_for_inference(result_df, checkpoint)
        with torch.no_grad():
            logits = model(torch.from_numpy(X).to(device))
            probs = torch.sigmoid(logits).cpu().numpy()

        iterator = tqdm(range(len(result_df)), total=len(result_df)) if verbose else range(len(result_df))
        important_labels = []
        for idx in iterator:
            row = result_df.iloc[idx]
            labels_dict = _row_labels_from_buckets(row)
            top_idx = np.argsort(probs[idx])[-k:][::-1].tolist()
            important_labels.append(_top_indices_to_labels(top_idx, labels_dict, k=k))
        result_df["important_labels"] = important_labels
    except Exception as exc:
        print(f"Warning: Could not rank features with MLP: {exc}")
        result_df["important_labels"] = [[] for _ in range(len(result_df))]

    return result_df
