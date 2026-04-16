#!/usr/bin/env python3
"""
Точка входа для обучения v2 (XGBoost).

Запуск:
    python -m training_v2.run
"""

import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import optuna
import training_v2.kaggle_optuna as kopt
from training_v2.kaggle_optuna import (
    prepare_data_kaggle,
    objective_xgboost,
    train_best_model,
    save_checkpoint,
)


def main():
    features_csv = "features_dataset_bucket.csv"
    annotations_csv = "annotations.csv"
    checkpoint_path = "training_v2/checkpoints/xgb_importance.pkl"
    n_trials = 30

    print("=" * 70)
    print("  XGBoost: 24 бинарных модели + Optuna")
    print("=" * 70)

    X_train, X_val, y_train, y_val, num_cols, cat_cols, all_features, fitted_enc = (
        prepare_data_kaggle(features_csv, annotations_csv)
    )

    # Set global variables for Optuna objective
    kopt._X_train = X_train
    kopt._X_val = X_val
    kopt._y_train = y_train
    kopt._y_val = y_val
    kopt._cat_cols = cat_cols

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_xgboost, n_trials=n_trials)

    print(f"\n{'=' * 70}")
    print("  ЛУЧШИЕ ПАРАМЕТРЫ (XGBoost)")
    print("=" * 70)
    print(study.best_params)
    print(f"Лучший Score: {study.best_value:.4f}")

    # Train final model with best params and save
    print(f"\n{'=' * 70}")
    print("  Обучение финальной модели с лучшими параметрами...")
    print("=" * 70)
    best_models, best_score = train_best_model(X_train, X_val, y_train, y_val, study.best_params)
    print(f"Финальный Score: {best_score:.4f}")

    save_checkpoint(
        checkpoint_path, best_models, kopt._valid_target_indices,
        fitted_enc, all_features, study.best_params,
    )
    print(f"\nГотово. Чекпоинт: {checkpoint_path}")


if __name__ == "__main__":
    main()
