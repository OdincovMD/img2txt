#!/usr/bin/env python3
"""Entry point for training the XGBoost importance model."""

import argparse

from model.config import (
    DEFAULT_ANNOTATIONS_CSV,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FEATURES_CSV,
    DEFAULT_FEATURE_SET,
    OPTUNA_TRIALS,
)
from model.model import (
    load_checkpoint,
    optimize_xgboost,
    prepare_training_data,
    save_checkpoint,
    train_xgboost_models,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the XGBoost importance model")
    parser.add_argument("--features-csv", default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--annotations-csv", default=DEFAULT_ANNOTATIONS_CSV)
    parser.add_argument(
        "--feature-set",
        default=DEFAULT_FEATURE_SET,
        choices=["numeric_only", "selected_buckets", "all_buckets"],
    )
    parser.add_argument("--n-trials", type=int, default=OPTUNA_TRIALS)
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--skip-optuna", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    print("=" * 70)
    print("  XGBoost importance model")
    print("=" * 70)

    X_train, X_val, y_train, y_val, feature_columns, valid_target_indices = prepare_training_data(
        args.features_csv,
        args.annotations_csv,
        feature_set=args.feature_set,
    )
    if args.skip_optuna:
        try:
            checkpoint = load_checkpoint(args.checkpoint_path)
            best_params = checkpoint["params"]
        except Exception as exc:
            raise RuntimeError(
                f"Could not load params from checkpoint {args.checkpoint_path!r}. "
                "Run without --skip-optuna or pass a valid checkpoint."
            ) from exc
        print("\nSkipping Optuna, using params from checkpoint:")
        print(best_params)
    else:
        study = optimize_xgboost(
            X_train,
            X_val,
            y_train,
            y_val,
            valid_target_indices,
            args.n_trials,
        )

        print(f"\n{'=' * 70}")
        print("  BEST PARAMS")
        print("=" * 70)
        print(study.best_params)
        print(f"Best score: {study.best_value:.4f}")
        best_params = study.best_params

    models, score = train_xgboost_models(
        X_train,
        X_val,
        y_train,
        y_val,
        best_params,
        valid_target_indices,
    )
    print(f"Final score: {score:.4f}")

    save_checkpoint(
        args.checkpoint_path,
        models,
        valid_target_indices,
        feature_columns,
        best_params,
        feature_set=args.feature_set,
    )
    print(f"Checkpoint saved to: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
