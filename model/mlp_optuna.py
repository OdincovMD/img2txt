#!/usr/bin/env python3
"""Optuna search for the MLP importance model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

from model.config import (
    DEFAULT_ANNOTATIONS_CSV,
    DEFAULT_FEATURES_CSV,
    DEFAULT_MLP_CHECKPOINT_PATH,
    DEFAULT_MLP_FEATURE_SET,
    SEED,
)
from model.mlp import (
    build_mlp_feature_matrix,
    fit_feature_scaler,
    save_mlp_checkpoint,
    train_mlp_model,
    transform_features,
)
from model.model import FeatureSet, build_target_vector, parse_expert_labels


REQUIRED_MLP_PARAMS = [
    "batch_size",
    "learning_rate",
    "weight_decay",
    "dropout",
    "max_epochs",
    "patience",
    "hidden_dims",
]


def suggest_mlp_param_from_spec(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    kind = spec.get("type")
    if kind == "float":
        return trial.suggest_float(
            name,
            spec["low"],
            spec["high"],
            log=bool(spec.get("log", False)),
            step=spec.get("step"),
        )
    if kind == "int":
        return trial.suggest_int(
            name,
            spec["low"],
            spec["high"],
            log=bool(spec.get("log", False)),
            step=int(spec.get("step", 1)),
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unsupported Optuna spec for {name!r}: {spec!r}")


def normalize_mlp_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = params.copy()
    if "hidden_dims" not in normalized and {"hidden1", "hidden2"}.issubset(normalized):
        normalized["hidden_dims"] = (normalized["hidden1"], normalized["hidden2"])
    if "hidden_dims" in normalized:
        hidden_dims = tuple(normalized["hidden_dims"])
        if len(hidden_dims) != 2:
            raise ValueError(f"hidden_dims must contain exactly 2 values, got {hidden_dims!r}.")
        normalized["hidden_dims"] = (int(hidden_dims[0]), int(hidden_dims[1]))
    missing = [name for name in REQUIRED_MLP_PARAMS if name not in normalized]
    if missing:
        raise ValueError(f"Missing required MLP params: {', '.join(missing)}")
    return normalized


def suggest_mlp_params(
    trial: optuna.Trial,
    input_dim: int,
    optuna_kwargs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if optuna_kwargs is not None:
        params = {name: suggest_mlp_param_from_spec(trial, name, spec) for name, spec in optuna_kwargs.items()}
        return normalize_mlp_params(params)

    wide_upper = 256 if input_dim <= 64 else 384
    hidden1 = trial.suggest_categorical("hidden1", [64, 128, 192, 256, wide_upper])
    hidden2_upper = max(32, min(hidden1, 192))
    hidden2 = trial.suggest_categorical(
        "hidden2",
        sorted({32, 64, 96, 128, hidden2_upper}),
    )
    hidden2 = min(hidden2, hidden1)
    return {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "max_epochs": trial.suggest_int("max_epochs", 40, 100),
        "patience": trial.suggest_int("patience", 8, 20),
        "hidden_dims": (hidden1, hidden2),
    }


def _prepare_dataset(
    features_csv: str,
    annotations_csv: str,
    feature_set: FeatureSet,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    feature_df = pd.read_csv(features_csv)
    annotations_df = pd.read_csv(annotations_csv)
    merged = feature_df.merge(annotations_df[["image_path", "important_labels"]], on="image_path", how="inner")
    X_full, numeric_columns = build_mlp_feature_matrix(merged, feature_set=feature_set)
    if X_full.empty:
        raise ValueError("No MLP features found in feature table.")
    targets = np.stack([build_target_vector(parse_expert_labels(value)) for value in merged["important_labels"]]).astype(
        np.float32
    )
    return X_full, targets, numeric_columns


def optimize_mlp(
    features_csv: str = DEFAULT_FEATURES_CSV,
    annotations_csv: str = DEFAULT_ANNOTATIONS_CSV,
    feature_set: FeatureSet = DEFAULT_MLP_FEATURE_SET,
    n_trials: int = 20,
    n_splits: int = 5,
    verbose: bool = True,
    kwargs: dict[str, dict[str, Any]] | None = None,
) -> tuple[optuna.Study, pd.DataFrame]:
    X_full, targets, numeric_columns = _prepare_dataset(features_csv, annotations_csv, feature_set)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_mlp_params(trial, input_dim=X_full.shape[1], optuna_kwargs=kwargs)
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
            _, metadata = train_mlp_model(
                X_train_np,
                X_val_np,
                y_train,
                y_val,
                feature_columns=feature_columns,
                numeric_columns=numeric_columns,
                scaler=scaler,
                feature_set=feature_set,
                batch_size=params["batch_size"],
                max_epochs=params["max_epochs"],
                patience=params["patience"],
                learning_rate=params["learning_rate"],
                weight_decay=params["weight_decay"],
                dropout=params["dropout"],
                hidden_dims_override=params["hidden_dims"],
            )
            fold_scores.append(float(metadata["best_score"]))
            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(objective, n_trials=n_trials)

    trials_df = study.trials_dataframe().sort_values("value", ascending=False).reset_index(drop=True)
    if verbose:
        print(f"Best score: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
    return study, trials_df


def train_final_mlp_from_study(
    study: optuna.Study,
    features_csv: str = DEFAULT_FEATURES_CSV,
    annotations_csv: str = DEFAULT_ANNOTATIONS_CSV,
    checkpoint_path: str | Path = DEFAULT_MLP_CHECKPOINT_PATH,
    feature_set: FeatureSet = DEFAULT_MLP_FEATURE_SET,
) -> dict[str, Any]:
    X_full, targets, numeric_columns = _prepare_dataset(features_csv, annotations_csv, feature_set)
    scaler = fit_feature_scaler(X_full, numeric_columns)
    feature_columns = X_full.columns.tolist()
    X_full_np = transform_features(X_full, scaler, numeric_columns)
    params = normalize_mlp_params(study.best_params)
    model, metadata = train_mlp_model(
        X_full_np,
        X_full_np,
        targets,
        targets,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        scaler=scaler,
        feature_set=feature_set,
        batch_size=params["batch_size"],
        max_epochs=params["max_epochs"],
        patience=params["patience"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        dropout=params["dropout"],
        hidden_dims_override=params["hidden_dims"],
    )
    metadata["optuna_best_params"] = study.best_params
    save_mlp_checkpoint(checkpoint_path, model, metadata)
    return metadata


def mlp(
    features_csv: str = DEFAULT_FEATURES_CSV,
    annotations_csv: str = DEFAULT_ANNOTATIONS_CSV,
    feature_set: FeatureSet = DEFAULT_MLP_FEATURE_SET,
    n_trials: int = 20,
    n_splits: int = 5,
    checkpoint_path: str | Path = DEFAULT_MLP_CHECKPOINT_PATH,
    skip_final_train: bool = False,
    kwargs: dict[str, dict[str, Any]] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    study, trials_df = optimize_mlp(
        features_csv=features_csv,
        annotations_csv=annotations_csv,
        feature_set=feature_set,
        n_trials=n_trials,
        n_splits=n_splits,
        verbose=verbose,
        kwargs=kwargs,
    )
    if verbose:
        print("\nTop trials:")
        print(trials_df.head(10).to_string(index=False))

    metadata = None
    if not skip_final_train:
        metadata = train_final_mlp_from_study(
            study,
            features_csv=features_csv,
            annotations_csv=annotations_csv,
            checkpoint_path=checkpoint_path,
            feature_set=feature_set,
        )
        if verbose:
            print(f"\nFinal checkpoint saved to: {checkpoint_path}")
            print(f"Final score on full data reference: {metadata['best_score']:.4f}")

    return {
        "study": study,
        "trials_df": trials_df,
        "metadata": metadata,
        "checkpoint_path": str(checkpoint_path),
        "best_params": study.best_params,
        "best_score": study.best_value,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna search for MLP importance model")
    parser.add_argument("--features-csv", default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--annotations-csv", default=DEFAULT_ANNOTATIONS_CSV)
    parser.add_argument("--feature-set", default=DEFAULT_MLP_FEATURE_SET, choices=["numeric_only", "all_buckets"])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_MLP_CHECKPOINT_PATH))
    parser.add_argument("--skip-final-train", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    mlp(
        features_csv=args.features_csv,
        annotations_csv=args.annotations_csv,
        feature_set=args.feature_set,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        checkpoint_path=args.checkpoint_path,
        skip_final_train=args.skip_final_train,
        verbose=True,
    )


if __name__ == "__main__":
    main()
