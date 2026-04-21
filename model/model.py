"""Training and inference utilities for the XGBoost importance model."""

import json
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split

from bucketing.schema import bucket_column_name
from extraction.derived_features import COMPOSITE_FEATURE_COLUMNS
from extraction.feature_schema import FEATURE_COLUMNS
from model.config import (
    DEFAULT_XGB_MODEL_TYPE,
    DEFAULT_FEATURE_SET,
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


def scores_to_logits(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(scores, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def apply_calibration_bias(scores: np.ndarray, calibration_bias: list[float] | np.ndarray | None) -> np.ndarray:
    if calibration_bias is None:
        return scores
    bias = np.asarray(calibration_bias, dtype=np.float32)
    if bias.shape[0] != scores.shape[1]:
        raise ValueError(f"Calibration bias has {bias.shape[0]} labels, expected {scores.shape[1]}.")
    return scores_to_logits(scores) + bias


def build_full_target_matrix(y: np.ndarray, valid_target_indices: list[int]) -> np.ndarray:
    full_y = np.zeros((len(y), NUM_LABELS), dtype=np.float32)
    for model_idx, target_idx in enumerate(valid_target_indices):
        full_y[:, target_idx] = y[:, model_idx]
    return full_y


def evaluate_score_matrix(scores: np.ndarray, full_y: np.ndarray, k: int = 10) -> dict[str, float]:
    precision_values, recall_values = [], []
    for row_idx in range(len(scores)):
        true_indices = np.where(full_y[row_idx] > 0)[0].tolist()
        if not true_indices:
            continue
        p_at_k, r_at_k = calculate_precision_recall_at_k(scores[row_idx], true_indices, k=k)
        precision_values.append(p_at_k)
        recall_values.append(r_at_k)

    if not precision_values:
        return {"score": 0.0, "precision": 0.0, "recall": 0.0}

    precision = float(np.mean(precision_values))
    recall = float(np.mean(recall_values))
    return {"score": (precision + recall) / 2, "precision": precision, "recall": recall}


def calculate_topk_distribution(scores: np.ndarray, k: int = 10) -> np.ndarray:
    counts = np.zeros(scores.shape[1], dtype=np.float32)
    for row_idx in range(len(scores)):
        top_indices = np.argsort(scores[row_idx])[-k:]
        counts[top_indices] += 1
    return counts / max(len(scores) * k, 1)


def calculate_label_report(scores: np.ndarray, full_y: np.ndarray, k: int = 10) -> list[dict[str, Any]]:
    selected_counts = np.zeros(scores.shape[1], dtype=np.float32)
    hit_counts = np.zeros(scores.shape[1], dtype=np.float32)
    positive_counts = full_y.sum(axis=0).astype(np.float32)

    for row_idx in range(len(scores)):
        top_indices = np.argsort(scores[row_idx])[-k:]
        true_indices = set(np.where(full_y[row_idx] > 0)[0].tolist())
        for target_idx in top_indices:
            selected_counts[target_idx] += 1
            if target_idx in true_indices:
                hit_counts[target_idx] += 1

    total_positive = max(float(positive_counts.sum()), 1.0)
    total_selected = max(float(len(scores) * k), 1.0)
    report = []
    for target_idx, label in enumerate(LABEL_NAMES):
        positives = float(positive_counts[target_idx])
        selected = float(selected_counts[target_idx])
        hits = float(hit_counts[target_idx])
        true_share = positives / total_positive
        selected_share = selected / total_selected
        report.append(
            {
                "label": label,
                "positives": int(positives),
                "selected": int(selected),
                "hits": int(hits),
                "true_share": true_share,
                "selected_share": selected_share,
                "share_delta": selected_share - true_share,
                "recall_at_k": hits / positives if positives else 0.0,
                "precision_when_selected": hits / selected if selected else 0.0,
            }
        )
    return report


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    ordered = [*FEATURE_COLUMNS, *COMPOSITE_FEATURE_COLUMNS]
    return [col for col in ordered if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]


FeatureSet = Literal["numeric_only", "selected_buckets", "all_buckets"]
XgbModelType = Literal["xgb", "xgb_classifier_chain"]


def get_bucket_feature_columns(df: pd.DataFrame, feature_set: FeatureSet = DEFAULT_FEATURE_SET) -> list[str]:
    all_bucket_cols = sorted(col for col in df.columns if col.startswith(bucket_column_name("")))
    if feature_set == "numeric_only":
        return []
    if feature_set == "all_buckets":
        return all_bucket_cols
    ordered = [bucket_column_name(label) for label in LABEL_NAMES]
    return [col for col in ordered if col in df.columns]


def build_model_matrix(df: pd.DataFrame, feature_set: FeatureSet = DEFAULT_FEATURE_SET) -> pd.DataFrame:
    numeric_cols = get_numeric_feature_columns(df)
    bucket_cols = get_bucket_feature_columns(df, feature_set=feature_set)

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
    X = build_model_matrix(feat_df, feature_set=checkpoint.get("feature_set", DEFAULT_FEATURE_SET))
    missing_cols = [col for col in checkpoint["feature_columns"] if col not in X.columns]
    if missing_cols:
        missing_df = pd.DataFrame(0.0, index=X.index, columns=missing_cols, dtype=np.float32)
        X = pd.concat([X, missing_df], axis=1)
    return X[checkpoint["feature_columns"]]


def prepare_training_data(features_csv: str, annotations_csv: str, feature_set: FeatureSet = DEFAULT_FEATURE_SET):
    feature_df = pd.read_csv(features_csv)
    annotations_df = pd.read_csv(annotations_csv)

    X_full = build_model_matrix(feature_df, feature_set=feature_set)
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


def get_xgb_model_type(checkpoint: dict[str, Any]) -> XgbModelType:
    model_type = checkpoint.get("model_type", DEFAULT_XGB_MODEL_TYPE)
    if model_type not in ("xgb", "xgb_classifier_chain"):
        return DEFAULT_XGB_MODEL_TYPE
    return model_type


def get_chain_feature_name(target_idx: int) -> str:
    return f"chain_pred__{LABEL_NAMES[target_idx]}"


def append_chain_features(
    X: pd.DataFrame,
    chain_target_indices: list[int],
    chain_predictions: list[np.ndarray],
) -> pd.DataFrame:
    if not chain_target_indices:
        return X
    frame = X.copy()
    for target_idx, preds in zip(chain_target_indices, chain_predictions):
        frame[get_chain_feature_name(target_idx)] = np.asarray(preds, dtype=np.float32)
    return frame


def build_xgb_training_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
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
        "device": get_xgb_device(),
        "seed": SEED,
        "verbosity": 0,
    }


def fit_xgb_booster(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict[str, Any],
) -> xgb.Booster:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    return xgb.train(
        build_xgb_training_params(params),
        dtrain,
        num_boost_round=params["n_estimators"],
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )


def build_oof_predictions(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    class_counts = np.bincount(y_train.astype(np.int32), minlength=2)
    min_class_count = int(class_counts.min()) if len(class_counts) == 2 else 0
    if min_class_count < 2:
        return np.full(len(X_train), float(y_train.mean()), dtype=np.float32)

    n_splits = min(5, min_class_count)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X_train), dtype=np.float32)
    X_train_reset = X_train.reset_index(drop=True)

    for fold_train_idx, fold_val_idx in splitter.split(X_train_reset, y_train.astype(np.int32)):
        booster = fit_xgb_booster(
            X_train_reset.iloc[fold_train_idx],
            y_train[fold_train_idx],
            X_train_reset.iloc[fold_val_idx],
            y_train[fold_val_idx],
            params,
        )
        oof_preds[fold_val_idx] = booster.predict(xgb.DMatrix(X_train_reset.iloc[fold_val_idx]), output_margin=False)
    return oof_preds


def predict_xgboost_scores(
    models: list[xgb.Booster],
    X: pd.DataFrame,
    valid_target_indices: list[int],
    model_type: XgbModelType = DEFAULT_XGB_MODEL_TYPE,
    chain_order: list[int] | None = None,
) -> np.ndarray:
    full_preds = np.zeros((len(X), NUM_LABELS), dtype=np.float32)
    if model_type == "xgb_classifier_chain":
        active_chain = list(chain_order or valid_target_indices)
        previous_preds: list[np.ndarray] = []
        for model_idx, target_idx in enumerate(active_chain):
            chain_features = append_chain_features(X, active_chain[:model_idx], previous_preds)
            preds = models[model_idx].predict(xgb.DMatrix(chain_features), output_margin=False)
            full_preds[:, target_idx] = preds
            previous_preds.append(preds.astype(np.float32))
        return full_preds

    dmatrix = xgb.DMatrix(X)
    for model_idx, target_idx in enumerate(valid_target_indices):
        full_preds[:, target_idx] = models[model_idx].predict(dmatrix, output_margin=False)
    return full_preds


def evaluate_models(
    models: list[xgb.Booster],
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    valid_target_indices: list[int],
    model_type: XgbModelType = DEFAULT_XGB_MODEL_TYPE,
    chain_order: list[int] | None = None,
) -> float:
    full_preds = predict_xgboost_scores(
        models,
        X_val,
        valid_target_indices,
        model_type=model_type,
        chain_order=chain_order,
    )
    full_y = build_full_target_matrix(y_val, valid_target_indices)

    return evaluate_score_matrix(full_preds, full_y, k=10)["score"]


def calibrate_xgboost_label_bias(
    models: list[xgb.Booster],
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    valid_target_indices: list[int],
    model_type: XgbModelType = DEFAULT_XGB_MODEL_TYPE,
    chain_order: list[int] | None = None,
    k: int = 10,
    alpha_max: float = 2.0,
    steps: int = 81,
) -> dict[str, Any]:
    full_preds = predict_xgboost_scores(
        models,
        X_val,
        valid_target_indices,
        model_type=model_type,
        chain_order=chain_order,
    )
    full_y = build_full_target_matrix(y_val, valid_target_indices)
    base_scores = scores_to_logits(full_preds)

    true_share = full_y.sum(axis=0)
    true_share = true_share / max(float(true_share.sum()), 1.0)
    pred_share = calculate_topk_distribution(base_scores, k=k)
    base_bias = np.log((true_share + 1e-6) / (pred_share + 1e-6))
    base_bias = np.clip(base_bias, -3.0, 3.0).astype(np.float32)

    baseline_metrics = evaluate_score_matrix(base_scores, full_y, k=k)
    baseline_report = calculate_label_report(base_scores, full_y, k=k)
    best_alpha = 0.0
    best_bias = np.zeros(NUM_LABELS, dtype=np.float32)
    best_metrics = baseline_metrics

    for alpha in np.linspace(0.0, alpha_max, steps):
        candidate_bias = (alpha * base_bias).astype(np.float32)
        metrics = evaluate_score_matrix(base_scores + candidate_bias, full_y, k=k)
        if metrics["score"] > best_metrics["score"]:
            best_alpha = float(alpha)
            best_bias = candidate_bias
            best_metrics = metrics

    calibrated_share = calculate_topk_distribution(base_scores + best_bias, k=k)
    calibrated_report = calculate_label_report(base_scores + best_bias, full_y, k=k)
    return {
        "bias": best_bias.tolist(),
        "alpha": best_alpha,
        "k": k,
        "baseline_metrics": baseline_metrics,
        "calibrated_metrics": best_metrics,
        "baseline_report": baseline_report,
        "calibrated_report": calibrated_report,
        "true_share": true_share.astype(float).tolist(),
        "pred_share_before": pred_share.astype(float).tolist(),
        "pred_share_after": calibrated_share.astype(float).tolist(),
    }


def train_xgboost_models(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    valid_target_indices: list[int],
) -> tuple[list[xgb.Booster], float, dict[str, Any]]:
    return train_xgboost_models_by_type(
        X_train,
        X_val,
        y_train,
        y_val,
        params,
        valid_target_indices,
        model_type=DEFAULT_XGB_MODEL_TYPE,
    )


def train_xgboost_models_by_type(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    valid_target_indices: list[int],
    model_type: XgbModelType = DEFAULT_XGB_MODEL_TYPE,
) -> tuple[list[xgb.Booster], float, dict[str, Any]]:
    if model_type == "xgb_classifier_chain":
        return train_xgb_classifier_chain(X_train, X_val, y_train, y_val, params, valid_target_indices)

    models = []

    for target_idx in range(len(valid_target_indices)):
        booster = fit_xgb_booster(X_train, y_train[:, target_idx], X_val, y_val[:, target_idx], params)
        models.append(booster)

    score = evaluate_models(models, X_val, y_val, valid_target_indices, model_type=model_type)
    return models, score, {"chain_order": list(valid_target_indices)}


def train_xgb_classifier_chain(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    valid_target_indices: list[int],
) -> tuple[list[xgb.Booster], float, dict[str, Any]]:
    chain_order = list(valid_target_indices)
    models: list[xgb.Booster] = []
    train_chain_predictions: list[np.ndarray] = []
    val_chain_predictions: list[np.ndarray] = []

    for model_idx, target_idx in enumerate(chain_order):
        previous_targets = chain_order[:model_idx]
        X_train_step = append_chain_features(X_train, previous_targets, train_chain_predictions)
        X_val_step = append_chain_features(X_val, previous_targets, val_chain_predictions)
        y_train_target = y_train[:, model_idx]
        y_val_target = y_val[:, model_idx]

        train_oof_preds = build_oof_predictions(X_train_step, y_train_target, params)
        booster = fit_xgb_booster(X_train_step, y_train_target, X_val_step, y_val_target, params)
        val_preds = booster.predict(xgb.DMatrix(X_val_step), output_margin=False).astype(np.float32)

        models.append(booster)
        train_chain_predictions.append(train_oof_preds)
        val_chain_predictions.append(val_preds)

    score = evaluate_models(
        models,
        X_val,
        y_val,
        valid_target_indices,
        model_type="xgb_classifier_chain",
        chain_order=chain_order,
    )
    return models, score, {"chain_order": chain_order}


def suggest_xgb_params(trial: optuna.Trial, feature_set: FeatureSet = DEFAULT_FEATURE_SET) -> dict[str, Any]:
    shared_params = {
        "learning_rate": trial.suggest_float("learning_rate", 3e-3, 0.08, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 500, 1600),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 6.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
    }
    if feature_set == "numeric_only":
        return {
            **shared_params,
            "max_depth": trial.suggest_int("max_depth", 3, 6),
        }
    if feature_set == "all_buckets":
        return {
            **shared_params,
            "max_depth": trial.suggest_int("max_depth", 5, 9),
        }
    return {
        **shared_params,
        "max_depth": trial.suggest_int("max_depth", 4, 7),
    }


def optimize_xgboost(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    valid_target_indices: list[int],
    n_trials: int,
    feature_set: FeatureSet = DEFAULT_FEATURE_SET,
    model_type: XgbModelType = DEFAULT_XGB_MODEL_TYPE,
) -> optuna.study.Study:
    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial, feature_set=feature_set)
        _, score, _ = train_xgboost_models_by_type(
            X_train,
            X_val,
            y_train,
            y_val,
            params,
            valid_target_indices,
            model_type=model_type,
        )
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
    feature_set: FeatureSet = DEFAULT_FEATURE_SET,
    model_type: XgbModelType = DEFAULT_XGB_MODEL_TYPE,
    chain_order: list[int] | None = None,
    calibration_bias: list[float] | None = None,
    calibration_metadata: dict[str, Any] | None = None,
) -> None:
    serialized_models = [model.save_raw() for model in models]
    checkpoint = {
        "models_raw": serialized_models,
        "valid_target_indices": valid_target_indices,
        "feature_columns": feature_columns,
        "params": params,
        "feature_set": feature_set,
        "model_type": model_type,
        "chain_order": list(chain_order or valid_target_indices),
        "label_names": list(LABEL_NAMES),
        "num_labels": NUM_LABELS,
        "calibration_bias": calibration_bias,
        "calibration_metadata": calibration_metadata,
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
    checkpoint.setdefault("model_type", DEFAULT_XGB_MODEL_TYPE)
    if not checkpoint.get("feature_set"):
        checkpoint["feature_set"] = DEFAULT_FEATURE_SET
    checkpoint.setdefault("chain_order", list(checkpoint["valid_target_indices"]))
    checkpoint.setdefault("calibration_bias", None)
    checkpoint.setdefault("calibration_metadata", None)
    del checkpoint["models_raw"]
    return checkpoint
