"""Training and evaluation utilities for the MLP importance model."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model.config import (
    DEFAULT_ANNOTATIONS_CSV,
    DEFAULT_FEATURES_CSV,
    DEFAULT_MLP_CHECKPOINT_PATH,
    DEFAULT_MLP_FEATURE_SET,
    LABEL_NAMES,
    MLP_BATCH_SIZE,
    MLP_DROPOUT,
    MLP_LEARNING_RATE,
    MLP_MAX_EPOCHS,
    MLP_PATIENCE,
    MLP_WEIGHT_DECAY,
    NUM_LABELS,
    SEED,
    TRAIN_SIZE,
    get_torch_device,
)
from model.model import (
    FeatureSet,
    build_model_matrix,
    build_target_vector,
    calculate_precision_recall_at_k,
    parse_expert_labels,
)


class MLPImportanceModel(nn.Module):
    """Small MLP for multi-label importance ranking."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, int], dropout: float = MLP_DROPOUT):
        super().__init__()
        hidden1, hidden2 = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, NUM_LABELS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def infer_hidden_dims(input_dim: int) -> tuple[int, int]:
    if input_dim <= 64:
        return 128, 64
    return 256, 128


def compute_pos_weight(targets: np.ndarray) -> torch.Tensor:
    positives = targets.sum(axis=0)
    negatives = len(targets) - positives
    safe_positives = np.where(positives > 0, positives, 1.0)
    weights = negatives / safe_positives
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def build_mlp_feature_matrix(df: pd.DataFrame, feature_set: FeatureSet = DEFAULT_MLP_FEATURE_SET) -> tuple[pd.DataFrame, list[str]]:
    X = build_model_matrix(df, feature_set=feature_set)
    numeric_columns = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col]) and not col.startswith("bucket_")]
    return X, numeric_columns


def fit_feature_scaler(X_train: pd.DataFrame, numeric_columns: list[str]) -> StandardScaler | None:
    if not numeric_columns:
        return None
    scaler = StandardScaler()
    scaler.fit(X_train[numeric_columns].astype(np.float32))
    return scaler


def transform_features(
    X: pd.DataFrame,
    scaler: StandardScaler | None,
    numeric_columns: list[str],
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    frame = X.copy()
    if feature_columns is not None:
        missing_cols = [col for col in feature_columns if col not in frame.columns]
        if missing_cols:
            missing_df = pd.DataFrame(0.0, index=frame.index, columns=missing_cols, dtype=np.float32)
            frame = pd.concat([frame, missing_df], axis=1)
        frame = frame[feature_columns]
    if scaler is not None and numeric_columns:
        cols = [col for col in numeric_columns if col in frame.columns]
        if cols:
            frame.loc[:, cols] = scaler.transform(frame[cols].astype(np.float32))
    return frame.astype(np.float32).to_numpy(copy=True)


def prepare_mlp_training_data(
    features_csv: str = DEFAULT_FEATURES_CSV,
    annotations_csv: str = DEFAULT_ANNOTATIONS_CSV,
    feature_set: FeatureSet = DEFAULT_MLP_FEATURE_SET,
):
    feature_df = pd.read_csv(features_csv)
    annotations_df = pd.read_csv(annotations_csv)
    merged = feature_df.merge(annotations_df[["image_path", "important_labels"]], on="image_path", how="inner")

    X_full, numeric_columns = build_mlp_feature_matrix(merged, feature_set=feature_set)
    if X_full.empty:
        raise ValueError("No MLP features found in feature table.")

    y = np.stack([build_target_vector(parse_expert_labels(value)) for value in merged["important_labels"]]).astype(np.float32)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y,
        test_size=TRAIN_SIZE,
        random_state=SEED,
    )

    scaler = fit_feature_scaler(X_train, numeric_columns)
    X_train_np = transform_features(X_train, scaler, numeric_columns)
    X_val_np = transform_features(X_val, scaler, numeric_columns, feature_columns=X_train.columns.tolist())
    return X_train_np, X_val_np, y_train, y_val, X_train.columns.tolist(), numeric_columns, scaler


def build_data_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_mlp_model(model: nn.Module, X_val: np.ndarray, y_val: np.ndarray, device: str) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_val).to(device))
        preds = torch.sigmoid(logits).cpu().numpy()

    p10_values, r10_values = [], []
    for row_idx in range(len(preds)):
        true_indices = np.where(y_val[row_idx] > 0)[0].tolist()
        if not true_indices:
            continue
        p_at_k, r_at_k = calculate_precision_recall_at_k(preds[row_idx], true_indices, k=10)
        p10_values.append(p_at_k)
        r10_values.append(r_at_k)

    if not p10_values:
        return 0.0
    return (sum(p10_values) / len(p10_values) + sum(r10_values) / len(r10_values)) / 2


def train_mlp_model(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    feature_columns: list[str],
    numeric_columns: list[str],
    scaler: StandardScaler | None,
    feature_set: FeatureSet = DEFAULT_MLP_FEATURE_SET,
    batch_size: int = MLP_BATCH_SIZE,
    max_epochs: int = MLP_MAX_EPOCHS,
    patience: int = MLP_PATIENCE,
    learning_rate: float = MLP_LEARNING_RATE,
    weight_decay: float = MLP_WEIGHT_DECAY,
    dropout: float = MLP_DROPOUT,
    hidden_dims_override: tuple[int, int] | None = None,
) -> tuple[MLPImportanceModel, dict[str, Any]]:
    device = get_torch_device()
    hidden_dims = hidden_dims_override or infer_hidden_dims(X_train.shape[1])
    model = MLPImportanceModel(X_train.shape[1], hidden_dims, dropout=dropout).to(device)

    train_loader = build_data_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    pos_weight = compute_pos_weight(y_train).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_score = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    best_val_loss = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_X)

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_val).to(device))
            val_loss = criterion(val_logits, torch.from_numpy(y_val).to(device)).item()
        scheduler.step(val_loss)

        score = evaluate_mlp_model(model, X_val, y_val, device)
        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metadata = {
        "feature_columns": list(feature_columns),
        "numeric_feature_columns": list(numeric_columns),
        "feature_set": feature_set,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_val_loss": best_val_loss,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "device": device,
        "scaler": scaler,
    }
    return model, metadata


def save_mlp_checkpoint(
    path: str | Path,
    model: MLPImportanceModel,
    metadata: dict[str, Any],
) -> None:
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": metadata["feature_columns"] and len(metadata["feature_columns"]) or 0,
        "feature_columns": metadata["feature_columns"],
        "numeric_feature_columns": metadata["numeric_feature_columns"],
        "feature_set": metadata["feature_set"],
        "hidden_dims": metadata["hidden_dims"],
        "dropout": metadata["dropout"],
        "scaler": metadata["scaler"],
        "label_names": list(LABEL_NAMES),
        "best_score": metadata["best_score"],
        "best_epoch": metadata["best_epoch"],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_mlp_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location="cpu")
    model = MLPImportanceModel(
        checkpoint["input_dim"],
        tuple(checkpoint["hidden_dims"]),
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint["model"] = model
    return checkpoint


def cross_validate_mlp(
    features_csv: str = DEFAULT_FEATURES_CSV,
    annotations_csv: str = DEFAULT_ANNOTATIONS_CSV,
    feature_sets: tuple[FeatureSet, ...] = ("numeric_only", "all_buckets"),
    n_splits: int = 5,
    max_epochs: int = MLP_MAX_EPOCHS,
    verbose: bool = True,
) -> pd.DataFrame:
    feature_df = pd.read_csv(features_csv)
    annotations_df = pd.read_csv(annotations_csv)
    merged = feature_df.merge(annotations_df[["image_path", "important_labels"]], on="image_path", how="inner")
    targets = np.stack([build_target_vector(parse_expert_labels(value)) for value in merged["important_labels"]]).astype(np.float32)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows: list[dict[str, Any]] = []

    for feature_set in feature_sets:
        X_full, numeric_columns = build_mlp_feature_matrix(merged, feature_set=feature_set)
        fold_scores: list[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_full), start=1):
            X_train = X_full.iloc[train_idx].copy()
            X_val = X_full.iloc[val_idx].copy()
            y_train = targets[train_idx].copy()
            y_val = targets[val_idx].copy()
            scaler = fit_feature_scaler(X_train, numeric_columns)
            feature_columns = X_train.columns.tolist()
            X_train_np = transform_features(X_train, scaler, numeric_columns)
            X_val_np = transform_features(X_val, scaler, numeric_columns, feature_columns=feature_columns)
            model, metadata = train_mlp_model(
                X_train_np,
                X_val_np,
                y_train,
                y_val,
                feature_columns=feature_columns,
                numeric_columns=numeric_columns,
                scaler=scaler,
                feature_set=feature_set,
                max_epochs=max_epochs,
            )
            fold_scores.append(float(metadata["best_score"]))
            if verbose:
                print(
                    f"{feature_set} fold={fold_idx} score={metadata['best_score']:.4f} "
                    f"features={len(feature_columns)}"
                )

        rows.append(
            {
                "feature_set": feature_set,
                "feature_columns": len(X_full.columns),
                "mean_score": float(np.mean(fold_scores)),
                "std_score": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
                "fold_scores": fold_scores,
            }
        )

    return pd.DataFrame(rows).sort_values("mean_score", ascending=False).reset_index(drop=True)


def train_and_save_mlp_model(
    features_csv: str = DEFAULT_FEATURES_CSV,
    annotations_csv: str = DEFAULT_ANNOTATIONS_CSV,
    checkpoint_path: str | Path = DEFAULT_MLP_CHECKPOINT_PATH,
    feature_set: FeatureSet = DEFAULT_MLP_FEATURE_SET,
    max_epochs: int = MLP_MAX_EPOCHS,
) -> tuple[MLPImportanceModel, dict[str, Any]]:
    X_train, X_val, y_train, y_val, feature_columns, numeric_columns, scaler = prepare_mlp_training_data(
        features_csv=features_csv,
        annotations_csv=annotations_csv,
        feature_set=feature_set,
    )
    model, metadata = train_mlp_model(
        X_train,
        X_val,
        y_train,
        y_val,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        scaler=scaler,
        feature_set=feature_set,
        max_epochs=max_epochs,
    )
    save_mlp_checkpoint(checkpoint_path, model, metadata)
    return model, metadata
