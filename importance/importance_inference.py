"""
Step 3: Feature importance ranking (XGBoost).
Loads trained XGBoost checkpoint, predicts top-k important features per image.
Public API: rank_features_batch(df, ...) -> df with 'important_labels' column.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from bucketing.threshold_rules import row_to_labels
from config.importance_config import LABEL_NAMES, NUM_LABELS
from training.kaggle_optuna import build_features_for_inference, load_checkpoint

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def _top_indices_to_labels(
    top_indices: List[int],
    labels_dict: dict,
    k: int = 10,
) -> List[str]:
    """Convert model output indices to 'feature:value' strings."""
    result = []
    for i in top_indices:
        if i >= len(LABEL_NAMES):
            continue
        key = LABEL_NAMES[i]
        val = labels_dict.get(key)
        if isinstance(val, str):
            result.append(f"{key}:{val}")
        elif isinstance(val, dict):
            for subk, subv in val.items():
                if isinstance(subv, str):
                    result.append(f"{key}.{subk}:{subv}")
            if not val and key in labels_dict:
                result.append(f"{key}:—")
        if len(result) >= k:
            break
    return result[:k]


def _predict_row(
    row: pd.Series,
    checkpoint: dict,
    k: int = 10,
) -> List[str]:
    """Predict top-k important features for a single row."""
    labels_dict = row_to_labels(row.to_dict())

    # Build single-row DataFrame for feature engineering
    row_df = pd.DataFrame([row])
    X = build_features_for_inference(row_df, checkpoint)

    models = checkpoint["models"]
    valid_indices = checkpoint["valid_target_indices"]

    # Predict with each binary model
    full_preds = np.zeros(NUM_LABELS)
    dmat = xgb.DMatrix(X)
    for idx, col_i in enumerate(valid_indices):
        full_preds[col_i] = models[idx].predict(dmat)[0]

    top_idx = np.argsort(full_preds)[::-1][:k].tolist()
    return _top_indices_to_labels(top_idx, labels_dict, k=k)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_features_batch(
    df: pd.DataFrame,
    importance_model_path: Optional[Union[str, Path]] = None,
    k: int = 10,
    verbose: bool = True,
    **_kwargs,
) -> pd.DataFrame:
    """
    Step 3: Rank features by importance for all rows in df.

    Args:
        df: DataFrame with numeric feature columns.
        importance_model_path: Path to XGBoost checkpoint (.pkl).
            If None, 'important_labels' will be empty lists.
        k: Number of top features to select.
        verbose: Show progress bar.

    Returns:
        df with added column:
        - important_labels: list of k strings ["feature:value", ...]
    """
    df = df.copy()

    if importance_model_path is None:
        df["important_labels"] = [[] for _ in range(len(df))]
        return df

    if xgb is None:
        print("Warning: xgboost not installed, skipping importance ranking.")
        df["important_labels"] = [[] for _ in range(len(df))]
        return df

    try:
        checkpoint = load_checkpoint(str(importance_model_path))

        results = []
        iterator = tqdm(df.iterrows(), total=len(df)) if verbose else df.iterrows()
        for _, row in iterator:
            pred = _predict_row(row, checkpoint, k=k)
            results.append(pred)

        df["important_labels"] = results

    except Exception as e:
        print(f"Warning: Could not rank features: {e}")
        df["important_labels"] = [[] for _ in range(len(df))]

    return df
