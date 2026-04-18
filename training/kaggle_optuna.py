"""
XGBoost training pipeline: 24 binary classifiers + Optuna hyperparameter search.

Public API:
    prepare_data_kaggle(features_csv, annotations_csv) — load & engineer features
    train_best_model(X_train, y_train, X_val, y_val, params) — train with best params
    save_checkpoint(path, ...) / load_checkpoint(path) — persist/restore full pipeline
    build_features_for_inference(feat_df) — feature engineering for a new DataFrame
"""

import json
import os
import pickle

import category_encoders as ce
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from training.config import LABEL_NAMES, NUM_LABELS

# ----------------- КОНСТАНТЫ -----------------
SEED = 42
GLOBAL_LABELS = LABEL_NAMES


# ----------------- УТИЛИТЫ ДЛЯ МЕТРИК -----------------
def parse_expert_labels(raw: str):
    if pd.isna(raw) or raw is None:
        return []
    if isinstance(raw, str):
        try:
            arr = json.loads(raw)
        except json.JSONDecodeError:
            return [raw.strip()] if raw.strip() else []
    else:
        arr = list(raw)
    return [str(x).strip() for x in arr if x][:10]


def label_key(s: str) -> str:
    return s.split(":", 1)[0].strip() if ":" in s else s.strip()


def build_target_vector(expert_strings):
    vector = np.zeros(NUM_LABELS, dtype=np.float32)
    for s in expert_strings:
        k = label_key(s)
        if k in GLOBAL_LABELS:
            vector[GLOBAL_LABELS.index(k)] = 1.0
    return vector


def calculate_precision_recall_at_k(preds, true_indices, k=10):
    if len(true_indices) == 0:
        return 0.0, 0.0
    top_k_indices = np.argsort(preds)[-k:][::-1]
    intersection = len(set(top_k_indices).intersection(set(true_indices)))
    precision = intersection / k
    recall = intersection / len(true_indices)
    return precision, recall


# ----------------- FEATURE ENGINEERING -----------------
def _safe_col(df, col, default=0.0):
    """Safely get a column from df, filling NaN with default."""
    return df[col].fillna(default) if col in df.columns else pd.Series(default, index=df.index)


def _palette_variety(raw):
    """Compute palette variety from dominant_colors_lesion (list or JSON string)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return 1.0
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return 1.0
    colors = raw
    if not isinstance(colors, list) or len(colors) == 0:
        return 1.0
    hues = []
    for c in colors:
        if isinstance(c, (list, tuple)) and len(c) >= 1:
            hues.append(c[0] % 180)
        elif isinstance(c, (int, float)):
            hues.append(float(c) % 180)
    if len(hues) < 2:
        return min(len(colors), 4.0)
    hues = sorted(hues)
    n = 1
    for i in range(1, len(hues)):
        if abs(hues[i] - hues[i - 1]) > 15 and abs(hues[i] - hues[0]) > 15:
            n += 1
    return min(n, 6.0)


def compute_derived_features(df):
    """
    Compute 11 derived numeric features from raw columns.
    Mutates df in-place and returns list of numeric column names.
    Used both in training and inference.
    """
    df['asymmetry'] = (
        _safe_col(df, 'delta_H_center_periphery').abs()
        + _safe_col(df, 'delta_S_center_periphery').abs()
        + _safe_col(df, 'delta_V_center_periphery').abs()
        + _safe_col(df, 'delta_V_left_right').abs()
        + _safe_col(df, 'delta_S_left_right').abs()
        + _safe_col(df, 'delta_V_top_bottom').abs()
        + _safe_col(df, 'delta_S_top_bottom').abs()
    )
    df['borders'] = _safe_col(df, 'circularity')
    df['contrast'] = _safe_col(df, 'color_distance_deltaE')

    if 'dominant_colors_lesion' in df.columns:
        df['palette'] = df['dominant_colors_lesion'].apply(_palette_variety)
    else:
        df['palette'] = 1.0

    df['texture'] = _safe_col(df, 'glcm_homogeneity')
    df['elongation'] = (_safe_col(df, 'aspect_ratio') - 1).clip(lower=0) * 0.5 + _safe_col(df, 'eccentricity') * 0.5
    df['rim'] = _safe_col(df, 'delta_V_inner_rim').abs()
    df['color_homogeneity'] = (
        _safe_col(df, 'std_H_lesion') / 30
        + _safe_col(df, 'std_S_lesion') / 100
        + _safe_col(df, 'std_V_lesion') / 100
        + _safe_col(df, 'entropy_H_lesion') / 4
    )
    df['shape'] = _safe_col(df, 'solidity')
    df['dominant_hue'] = _safe_col(df, 'mean_H_lesion')
    df['structure_order'] = _safe_col(df, 'lbp_uniformity')

    num_cols = [c for c in GLOBAL_LABELS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    return num_cols


def extract_categorical_features(df):
    """Extract bucketed categorical features from labels_json column."""
    cat_cols = []
    if 'labels_json' in df.columns:
        parsed = df['labels_json'].apply(lambda x: json.loads(x) if isinstance(x, str) else {})
        for label in GLOBAL_LABELS:
            cat_name = f"cat_{label}"
            df[cat_name] = parsed.apply(lambda x: str(x.get(label, 'Unknown')))
            cat_cols.append(cat_name)
    return cat_cols


def build_features_for_inference(feat_df, checkpoint):
    """
    Build feature matrix for inference using fitted encoders from checkpoint.

    Args:
        feat_df: DataFrame with raw feature columns + labels_json
        checkpoint: dict loaded via load_checkpoint()

    Returns:
        X: DataFrame with engineered features, columns matching training order
    """
    df = feat_df.copy()
    num_cols = compute_derived_features(df)
    cat_cols = extract_categorical_features(df)

    df_num = df[num_cols].copy()
    df_num = df_num.fillna(0)

    te_complexity = checkpoint.get("te_complexity")
    te_label_encoders = checkpoint.get("te_label_encoders", {})
    feature_columns = checkpoint["feature_columns"]

    if te_complexity is not None and len(cat_cols) > 0:
        df_te_compl = te_complexity.transform(df[cat_cols])
        df_te_compl.columns = [f"te_compl_{c}" for c in cat_cols]

        te_label_dfs = []
        for label, te in te_label_encoders.items():
            cat_col = f"cat_{label}"
            if cat_col in cat_cols:
                enc = te.transform(df[[cat_col]])
                te_label_dfs.append(enc.rename(columns={cat_col: f"te_lbl_{label}"}))

        df_te_lbl = pd.concat(te_label_dfs, axis=1) if te_label_dfs else pd.DataFrame(index=df.index)

        # Interactions
        inter_dfs = []
        for label in GLOBAL_LABELS:
            te_col = f"te_compl_cat_{label}"
            if te_col in df_te_compl.columns and label in df_num.columns:
                inter_dfs.append(pd.Series(df_te_compl[te_col] * df_num[label], name=f"inter_{label}"))

        X_inter = pd.concat(inter_dfs, axis=1) if inter_dfs else pd.DataFrame()
        X = pd.concat([df_num, df_te_compl, df_te_lbl, X_inter], axis=1)
    else:
        X = df_num

    # Align columns to training order, fill missing with 0
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_columns]

    return X


# ----------------- ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ -----------------
def prepare_data_kaggle(features_csv: str, annotations_csv: str):
    """
    Load data, engineer features, split train/val, fit target encoders.

    Returns:
        X_train, X_val, y_train, y_val, num_cols, cat_cols, feature_columns,
        fitted_encoders (dict with te_complexity, te_label_encoders)
    """
    print("Загрузка данных...")
    feat_df = pd.read_csv(features_csv)
    ann_df = pd.read_csv(annotations_csv)

    num_cols = compute_derived_features(feat_df)
    cat_cols = extract_categorical_features(feat_df)

    # Merge features with annotations
    df = feat_df[['image_path'] + num_cols + cat_cols].merge(
        ann_df[['image_path', 'important_labels']], on='image_path', how='inner'
    )
    df = df.dropna(subset=num_cols, how='all').reset_index(drop=True)
    df[num_cols] = df[num_cols].fillna(0)

    # Build 24-dim target vectors
    targets = []
    for _, row in df.iterrows():
        strs = parse_expert_labels(row.get('important_labels'))
        targets.append(build_target_vector(strs))
    y = np.stack(targets).astype(np.float32)

    print(f"Извлечено числовых признаков: {len(num_cols)}")
    print(f"Извлечено категориальных (бакетов): {len(cat_cols)}")

    # Train / Val Split BEFORE Target Encoding (prevent data leak)
    df_train, df_val, y_train, y_val = train_test_split(
        df, y, test_size=0.2, random_state=SEED
    )

    df_num_train = df_train[num_cols].copy()
    df_num_val = df_val[num_cols].copy()

    # Target Encoding
    te_complexity = None
    te_label_encoders = {}

    y_complexity_train = y_train.sum(axis=1)

    if len(cat_cols) > 0:
        # Complexity TE
        te_complexity = ce.TargetEncoder(cols=cat_cols, smoothing=10.0)
        df_te_compl_train = te_complexity.fit_transform(df_train[cat_cols], y_complexity_train)
        df_te_compl_val = te_complexity.transform(df_val[cat_cols])
        df_te_compl_train.columns = [f"te_compl_{c}" for c in cat_cols]
        df_te_compl_val.columns = [f"te_compl_{c}" for c in cat_cols]

        # Per-label TE
        te_label_dfs_train, te_label_dfs_val = [], []
        for i, label in enumerate(GLOBAL_LABELS):
            cat_col = f"cat_{label}"
            if cat_col not in cat_cols or len(np.unique(y_train[:, i])) < 2:
                continue
            te = ce.TargetEncoder(cols=[cat_col], smoothing=20.0)
            enc_train = te.fit_transform(df_train[[cat_col]], y_train[:, i])
            enc_val = te.transform(df_val[[cat_col]])
            te_label_dfs_train.append(enc_train.rename(columns={cat_col: f"te_lbl_{label}"}))
            te_label_dfs_val.append(enc_val.rename(columns={cat_col: f"te_lbl_{label}"}))
            te_label_encoders[label] = te

        df_te_lbl_train = pd.concat(te_label_dfs_train, axis=1) if te_label_dfs_train else pd.DataFrame(index=df_train.index)
        df_te_lbl_val = pd.concat(te_label_dfs_val, axis=1) if te_label_dfs_val else pd.DataFrame(index=df_val.index)

        # Interactions: complexity_TE * numeric feature
        encoded_dfs_train, encoded_dfs_val = [], []
        for label in GLOBAL_LABELS:
            te_col = f"te_compl_cat_{label}"
            if te_col in df_te_compl_train.columns and label in df_num_train.columns:
                inter_col = f"inter_{label}"
                encoded_dfs_train.append(pd.Series(df_te_compl_train[te_col] * df_num_train[label], name=inter_col))
                encoded_dfs_val.append(pd.Series(df_te_compl_val[te_col] * df_num_val[label], name=inter_col))

        X_inter_train = pd.concat(encoded_dfs_train, axis=1) if encoded_dfs_train else pd.DataFrame()
        X_inter_val = pd.concat(encoded_dfs_val, axis=1) if encoded_dfs_val else pd.DataFrame()

        X_train = pd.concat([df_num_train, df_te_compl_train, df_te_lbl_train, X_inter_train], axis=1)
        X_val = pd.concat([df_num_val, df_te_compl_val, df_te_lbl_val, X_inter_val], axis=1)
    else:
        X_train, X_val = df_num_train, df_num_val

    print(f"Итоговая размерность: Трейн={X_train.shape}, Оценка={X_val.shape}")

    # Remove flat targets (all-zero labels with no positive examples)
    valid_cols = []
    for i in range(y_train.shape[1]):
        if len(np.unique(y_train[:, i])) > 1:
            valid_cols.append(i)

    if len(valid_cols) < y_train.shape[1]:
        print(f"Удалено плоских таргетов: {y_train.shape[1] - len(valid_cols)}")
        y_train = y_train[:, valid_cols]
        y_val = y_val[:, valid_cols]

    global _valid_target_indices
    _valid_target_indices = valid_cols

    fitted_encoders = {
        "te_complexity": te_complexity,
        "te_label_encoders": te_label_encoders,
    }

    return X_train, X_val, y_train, y_val, num_cols, cat_cols, X_train.columns.tolist(), fitted_encoders


# ----------------- TRAIN / SAVE / LOAD -----------------
def train_best_model(X_train, X_val, y_train, y_val, params):
    """
    Train 24 binary XGBoost models with given hyperparameters.

    Returns:
        models: list of xgb.Booster
        score: float (avg of P@10 and R@10)
    """
    models = []
    for idx in range(len(_valid_target_indices)):
        dtrain_i = xgb.DMatrix(X_train, label=y_train[:, idx])
        dval_i = xgb.DMatrix(X_val, label=y_val[:, idx])

        xgb_params = {
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
            "device": "cuda",
            "seed": SEED,
            "verbosity": 0,
        }

        model = xgb.train(
            xgb_params, dtrain_i,
            num_boost_round=params["n_estimators"],
            evals=[(dval_i, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        models.append(model)

    score = evaluate_xgb(models, X_val, y_val)
    return models, score


def save_checkpoint(path, models, valid_target_indices, fitted_encoders, feature_columns, params):
    """
    Save full XGBoost pipeline checkpoint (models + encoders + metadata).

    Checkpoint format (pickle):
        {
            "models": list of xgb.Booster (serialized as raw bytes),
            "valid_target_indices": list of int,
            "te_complexity": fitted TargetEncoder or None,
            "te_label_encoders": dict[str, fitted TargetEncoder],
            "feature_columns": list of str,
            "params": dict,
            "label_names": list of str,
            "num_labels": int,
        }
    """
    # Serialize XGBoost models as raw bytes (portable)
    serialized_models = []
    for m in models:
        raw = m.save_raw()
        serialized_models.append(raw)

    checkpoint = {
        "models_raw": serialized_models,
        "valid_target_indices": valid_target_indices,
        "te_complexity": fitted_encoders.get("te_complexity"),
        "te_label_encoders": fitted_encoders.get("te_label_encoders", {}),
        "feature_columns": feature_columns,
        "params": params,
        "label_names": list(LABEL_NAMES),
        "num_labels": NUM_LABELS,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path):
    """
    Load XGBoost pipeline checkpoint.

    Returns:
        dict with keys: models, valid_target_indices, te_complexity,
        te_label_encoders, feature_columns, params, label_names, num_labels
    """
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    # Deserialize XGBoost models
    models = []
    for raw in checkpoint["models_raw"]:
        m = xgb.Booster()
        m.load_model(bytearray(raw))
        models.append(m)

    checkpoint["models"] = models
    del checkpoint["models_raw"]
    return checkpoint


# ----------------- OPTUNA: XGBoost -----------------

# Global variables for Optuna objective
_X_train, _X_val, _y_train, _y_val, _cat_cols = None, None, None, None, None


def evaluate_xgb(models, X_val, y_val):
    """Evaluate ensemble of 24 binary XGBoost models."""
    full_preds = np.zeros((len(X_val), NUM_LABELS))
    full_y = np.zeros((len(X_val), NUM_LABELS))

    for idx, col_i in enumerate(_valid_target_indices):
        full_preds[:, col_i] = models[idx].predict(xgb.DMatrix(X_val), output_margin=False)
        full_y[:, col_i] = y_val[:, idx]

    p10_list, r10_list = [], []
    for i in range(len(full_preds)):
        true_indices = np.where(full_y[i] > 0)[0].tolist()
        if len(true_indices) == 0:
            continue
        p_at_k, r_at_k = calculate_precision_recall_at_k(full_preds[i], true_indices, k=10)
        p10_list.append(p_at_k)
        r10_list.append(r_at_k)

    return (sum(p10_list) / len(p10_list) + sum(r10_list) / len(r10_list)) / 2


def objective_xgboost(trial):
    params = {
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

    models, score = train_best_model(_X_train, _X_val, _y_train, _y_val, params)
    return score


if __name__ == "__main__":
    GLOBAL_CSV = "features_dataset_bucket.csv"
    GLOBAL_ANN = "annotations.csv"

    if os.path.exists(GLOBAL_CSV):
        _X_train, _X_val, _y_train, _y_val, num_cols, _cat_cols, all_features, fitted_enc = (
            prepare_data_kaggle(GLOBAL_CSV, GLOBAL_ANN)
        )

        # XGBoost: 24 binary models (each specialized for its label)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_xgboost, n_trials=30)

        print("\n=== ЛУЧШИЕ ПАРАМЕТРЫ (XGBoost) ===")
        print(study.best_params)
        print(f"Лучший Score: {study.best_value:.4f}")

        # Train final model with best params and save checkpoint
        best_models, best_score = train_best_model(
            _X_train, _X_val, _y_train, _y_val, study.best_params
        )
        save_checkpoint(
            "training/checkpoints/xgb_importance.pkl",
            best_models, _valid_target_indices, fitted_enc,
            all_features, study.best_params,
        )
