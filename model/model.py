"""Training and inference utilities for the XGBoost importance model."""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from bucketing.schema import bucket_column_name
from extraction.derived_features import COMPOSITE_FEATURE_COLUMNS
from extraction.feature_schema import FEATURE_COLUMNS
from model.config import (
    EARLY_STOPPING_ROUNDS,
    LABEL_NAMES,
    NUM_LABELS,
    SEED,
    TRAIN_SIZE,
    get_xgb_device,
)


def parse_expert_labels(raw: Any) -> list[str]:
    if pd.isna(raw) or raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [raw.strip()] if raw.strip() else []
    else:
        parsed = list(raw)
    return [str(item).strip() for item in parsed if item][:10]


def label_key(value: str) -> str:
    return value.split(":", 1)[0].strip() if ":" in value else value.strip()


def build_target_vector(expert_strings: list[str]) -> np.ndarray:
    vector = np.zeros(NUM_LABELS, dtype=np.float32)
    for item in expert_strings:
        key = label_key(item)
        if key in LABEL_NAMES:
            vector[LABEL_NAMES.index(key)] = 1.0
    return vector


def calculate_precision_recall_at_k(preds: np.ndarray, true_indices: list[int], k: int = 10) -> tuple[float, float]:
    if not true_indices:
        return 0.0, 0.0
    top_k_indices = np.argsort(preds)[-k:][::-1]
    intersection = len(set(top_k_indices).intersection(true_indices))
    return intersection / k, intersection / len(true_indices)


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    ordered = [*FEATURE_COLUMNS, *COMPOSITE_FEATURE_COLUMNS]
    return [col for col in ordered if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]


def get_bucket_feature_columns(df: pd.DataFrame) -> list[str]:
    ordered = [bucket_column_name(label) for label in LABEL_NAMES]
    return [col for col in ordered if col in df.columns]


def build_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = get_numeric_feature_columns(df)
    bucket_cols = get_bucket_feature_columns(df)

    frames = []
    if numeric_cols:
        frames.append(df[numeric_cols].fillna(0.0).astype(np.float32))
    if bucket_cols:
        bucket_df = df[bucket_cols].fillna("Unknown").astype(str)
        frames.append(pd.get_dummies(bucket_df, columns=bucket_cols, dtype=np.float32))

    if not frames:
        return pd.DataFrame(index=df.index)
    return pd.concat(frames, axis=1)


def build_features_for_inference(feat_df: pd.DataFrame, checkpoint: dict[str, Any]) -> pd.DataFrame:
    X = build_model_matrix(feat_df)
    for col in checkpoint["feature_columns"]:
        if col not in X.columns:
            X[col] = 0.0
    return X[checkpoint["feature_columns"]]


def prepare_training_data(features_csv: str, annotations_csv: str):
    feature_df = pd.read_csv(features_csv)
    annotations_df = pd.read_csv(annotations_csv)

    X_full = build_model_matrix(feature_df)
    if X_full.empty:
        raise ValueError("No model features found in feature table. Expected numeric features and/or bucket_* columns.")

    df = pd.concat([feature_df[["image_path"]], X_full], axis=1).merge(
        annotations_df[["image_path", "important_labels"]], on="image_path", how="inner"
    )
    df = df.reset_index(drop=True)

    targets = [build_target_vector(parse_expert_labels(value)) for value in df["important_labels"]]
    y = np.stack(targets).astype(np.float32)

    X = df[X_full.columns].copy()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TRAIN_SIZE, random_state=SEED
    )

    valid_target_indices = [idx for idx in range(y_train.shape[1]) if len(np.unique(y_train[:, idx])) > 1]
    if len(valid_target_indices) < y_train.shape[1]:
        y_train = y_train[:, valid_target_indices]
        y_val = y_val[:, valid_target_indices]
    if not valid_target_indices:
        raise ValueError("No trainable targets found in annotations.")

    return X_train, X_val, y_train, y_val, X.columns.tolist(), valid_target_indices


def evaluate_models(models: list[xgb.Booster], X_val: pd.DataFrame, y_val: np.ndarray, valid_target_indices: list[int]) -> float:
    full_preds = np.zeros((len(X_val), NUM_LABELS), dtype=np.float32)
    full_y = np.zeros((len(X_val), NUM_LABELS), dtype=np.float32)
    dval = xgb.DMatrix(X_val)

    for model_idx, target_idx in enumerate(valid_target_indices):
        full_preds[:, target_idx] = models[model_idx].predict(dval, output_margin=False)
        full_y[:, target_idx] = y_val[:, model_idx]

    p10_values, r10_values = [], []
    for row_idx in range(len(full_preds)):
        true_indices = np.where(full_y[row_idx] > 0)[0].tolist()
        if not true_indices:
            continue
        p_at_k, r_at_k = calculate_precision_recall_at_k(full_preds[row_idx], true_indices, k=10)
        p10_values.append(p_at_k)
        r10_values.append(r_at_k)

    if not p10_values:
        return 0.0
    return (sum(p10_values) / len(p10_values) + sum(r10_values) / len(r10_values)) / 2


def train_xgboost_models(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    valid_target_indices: list[int],
) -> tuple[list[xgb.Booster], float]:
    models = []
    xgb_device = get_xgb_device()

    for target_idx in range(len(valid_target_indices)):
        dtrain = xgb.DMatrix(X_train, label=y_train[:, target_idx])
        dval = xgb.DMatrix(X_val, label=y_val[:, target_idx])
        booster = xgb.train(
            {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": params["max_depth"],
                "learning_rate": params["learning_rate"],
                "reg_lambda": params["reg_lambda"],
                "reg_alpha": params["reg_alpha"],
                "subsample": params["subsample"],
                "colsample_bytree": params["colsample_bytree"],
                "min_child_weight": params["min_child_weight"],
                "gamma": params["gamma"],
                "max_bin": params["max_bin"],
                "tree_method": "hist",
                "device": xgb_device,
                "seed": SEED,
                "verbosity": 0,
            },
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dval, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        models.append(booster)

    return models, evaluate_models(models, X_val, y_val, valid_target_indices)


def suggest_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "max_depth": trial.suggest_int("max_depth", 5, 7),
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.04, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 800, 1400),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 4.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 0.85),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.95),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
        "gamma": trial.suggest_float("gamma", 2.5, 4.5),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256]),
    }


def optimize_xgboost(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    valid_target_indices: list[int],
    n_trials: int,
) -> optuna.study.Study:
    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial)
        _, score = train_xgboost_models(X_train, X_val, y_train, y_val, params, valid_target_indices)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study


def save_checkpoint(
    path: str | Path,
    models: list[xgb.Booster],
    valid_target_indices: list[int],
    feature_columns: list[str],
    params: dict[str, Any],
) -> None:
    serialized_models = [model.save_raw() for model in models]
    checkpoint = {
        "models_raw": serialized_models,
        "valid_target_indices": valid_target_indices,
        "feature_columns": feature_columns,
        "params": params,
        "label_names": list(LABEL_NAMES),
        "num_labels": NUM_LABELS,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        checkpoint = pickle.load(handle)

    models = []
    for raw in checkpoint["models_raw"]:
        model = xgb.Booster()
        model.load_model(bytearray(raw))
        models.append(model)

    checkpoint["models"] = models
    del checkpoint["models_raw"]
    return checkpoint
