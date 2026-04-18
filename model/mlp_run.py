#!/usr/bin/env python3
"""Entry point for training and validating the MLP importance model."""

import argparse

from model.config import (
    DEFAULT_ANNOTATIONS_CSV,
    DEFAULT_FEATURES_CSV,
    DEFAULT_MLP_CHECKPOINT_PATH,
    DEFAULT_MLP_FEATURE_SET,
    MLP_MAX_EPOCHS,
)
from model.mlp import cross_validate_mlp, train_and_save_mlp_model


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

    print("=" * 70)
    print("  MLP importance model")
    print("=" * 70)

    if not args.skip_cv:
        cv_results = cross_validate_mlp(
            features_csv=args.features_csv,
            annotations_csv=args.annotations_csv,
            feature_sets=tuple(args.cv_feature_sets),
            n_splits=args.cv_splits,
            max_epochs=args.max_epochs,
        )
        print("\nCV results:")
        print(cv_results.to_string(index=False))

    _, metadata = train_and_save_mlp_model(
        features_csv=args.features_csv,
        annotations_csv=args.annotations_csv,
        checkpoint_path=args.checkpoint_path,
        feature_set=args.feature_set,
        max_epochs=args.max_epochs,
    )
    print(f"\nBest validation score: {metadata['best_score']:.4f}")
    print(f"Checkpoint saved to: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
