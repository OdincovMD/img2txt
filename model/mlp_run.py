#!/usr/bin/env python3
"""Entry point for training and validating the MLP importance model."""

import argparse
from pathlib import Path
from typing import Any

from model.config import (
    DEFAULT_ANNOTATIONS_CSV,
    DEFAULT_FEATURES_CSV,
    DEFAULT_MLP_CHECKPOINT_PATH,
    DEFAULT_MLP_FEATURE_SET,
    MLP_MAX_EPOCHS,
)
from model.mlp import (
    cross_validate_mlp,
    prepare_mlp_training_data,
    save_mlp_checkpoint,
    train_mlp_model,
)


def mlp_run(
    features_csv: str = DEFAULT_FEATURES_CSV,
    annotations_csv: str = DEFAULT_ANNOTATIONS_CSV,
    checkpoint_path: str | Path = DEFAULT_MLP_CHECKPOINT_PATH,
    feature_set: str = DEFAULT_MLP_FEATURE_SET,
    cv_feature_sets: tuple[str, ...] = ("numeric_only", "all_buckets"),
    cv_splits: int = 5,
    max_epochs: int = MLP_MAX_EPOCHS,
    skip_cv: bool = False,
    kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    if verbose:
        print("=" * 70)
        print("  MLP importance model")
        print("=" * 70)

    cv_results = None
    if not skip_cv:
        cv_results = cross_validate_mlp(
            features_csv=features_csv,
            annotations_csv=annotations_csv,
            feature_sets=cv_feature_sets,
            n_splits=cv_splits,
            max_epochs=max_epochs,
            verbose=verbose,
        )
        if verbose:
            print("\nCV results:")
            print(cv_results.to_string(index=False))

    X_train, X_val, y_train, y_val, feature_columns, numeric_columns, scaler = prepare_mlp_training_data(
        features_csv=features_csv,
        annotations_csv=annotations_csv,
        feature_set=feature_set,
    )
    train_kwargs = {"max_epochs": max_epochs}
    if kwargs:
        train_kwargs.update(kwargs)
    model, metadata = train_mlp_model(
        X_train,
        X_val,
        y_train,
        y_val,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        scaler=scaler,
        feature_set=feature_set,
        **train_kwargs,
    )
    save_mlp_checkpoint(checkpoint_path, model, metadata)

    if verbose:
        print(f"\nBest validation score: {metadata['best_score']:.4f}")
        print(f"Checkpoint saved to: {checkpoint_path}")

    return {
        "model": model,
        "metadata": metadata,
        "cv_results": cv_results,
        "checkpoint_path": str(checkpoint_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and validate the MLP importance model")
    parser.add_argument("--features-csv", default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--annotations-csv", default=DEFAULT_ANNOTATIONS_CSV)
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_MLP_CHECKPOINT_PATH))
    parser.add_argument("--feature-set", default=DEFAULT_MLP_FEATURE_SET, choices=["numeric_only", "all_buckets"])
    parser.add_argument(
        "--cv-feature-sets",
        nargs="*",
        default=["numeric_only", "all_buckets"],
        choices=["numeric_only", "all_buckets"],
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=MLP_MAX_EPOCHS)
    parser.add_argument("--skip-cv", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    mlp_run(
        features_csv=args.features_csv,
        annotations_csv=args.annotations_csv,
        checkpoint_path=args.checkpoint_path,
        feature_set=args.feature_set,
        cv_feature_sets=tuple(args.cv_feature_sets),
        cv_splits=args.cv_splits,
        max_epochs=args.max_epochs,
        skip_cv=args.skip_cv,
        verbose=True,
    )


if __name__ == "__main__":
    main()
