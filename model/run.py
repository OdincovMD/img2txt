#!/usr/bin/env python3
"""Entry point for training the XGBoost importance model."""

from model.config import (
    DEFAULT_ANNOTATIONS_CSV,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FEATURES_CSV,
    OPTUNA_TRIALS,
)
from model.model import (
    optimize_xgboost,
    prepare_training_data,
    save_checkpoint,
    train_xgboost_models,
)


def main() -> None:
    print("=" * 70)
    print("  XGBoost importance model")
    print("=" * 70)

    X_train, X_val, y_train, y_val, feature_columns, valid_target_indices = prepare_training_data(
        DEFAULT_FEATURES_CSV,
        DEFAULT_ANNOTATIONS_CSV,
    )
    study = optimize_xgboost(
        X_train,
        X_val,
        y_train,
        y_val,
        valid_target_indices,
        OPTUNA_TRIALS,
    )

    print(f"\n{'=' * 70}")
    print("  BEST PARAMS")
    print("=" * 70)
    print(study.best_params)
    print(f"Best score: {study.best_value:.4f}")

    models, score = train_xgboost_models(
        X_train,
        X_val,
        y_train,
        y_val,
        study.best_params,
        valid_target_indices,
    )
    print(f"Final score: {score:.4f}")

    save_checkpoint(
        DEFAULT_CHECKPOINT_PATH,
        models,
        valid_target_indices,
        feature_columns,
        study.best_params,
    )
    print(f"Checkpoint saved to: {DEFAULT_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
