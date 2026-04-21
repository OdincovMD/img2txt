#!/usr/bin/env python3
"""Entry point for training the XGBoost importance model."""

import argparse

from model.config import (
    DEFAULT_ANNOTATIONS_CSV,
    DEFAULT_FEATURES_CSV,
    DEFAULT_FEATURE_SET,
    DEFAULT_XGB_MODEL_TYPE,
    OPTUNA_TRIALS,
    get_default_xgb_checkpoint_path,
)
from model.model import (
    calibrate_xgboost_label_bias,
    load_checkpoint,
    optimize_xgboost,
    prepare_training_data,
    save_checkpoint,
    train_xgboost_models_by_type,
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
    parser.add_argument(
        "--model-type",
        default=DEFAULT_XGB_MODEL_TYPE,
        choices=["xgb", "xgb_classifier_chain"],
    )
    parser.add_argument("--n-trials", type=int, default=OPTUNA_TRIALS)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--skip-optuna", action="store_true")
    parser.add_argument("--calibrate-label-bias", action="store_true")
    parser.add_argument("--calibration-k", type=int, default=10)
    parser.add_argument("--calibration-alpha-max", type=float, default=2.0)
    parser.add_argument("--calibration-steps", type=int, default=81)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    print("=" * 70)
    print("  XGBoost importance model")
    print("=" * 70)

    checkpoint_path = args.checkpoint_path or str(get_default_xgb_checkpoint_path(args.feature_set, args.model_type))

    X_train, X_val, y_train, y_val, feature_columns, valid_target_indices = prepare_training_data(
        args.features_csv,
        args.annotations_csv,
        feature_set=args.feature_set,
    )
    if args.skip_optuna:
        try:
            checkpoint = load_checkpoint(checkpoint_path)
            checkpoint_feature_set = checkpoint.get("feature_set", DEFAULT_FEATURE_SET)
            if checkpoint_feature_set != args.feature_set:
                raise RuntimeError(
                    f"Checkpoint feature_set={checkpoint_feature_set!r} does not match "
                    f"requested feature_set={args.feature_set!r}."
                )
            checkpoint_model_type = checkpoint.get("model_type", DEFAULT_XGB_MODEL_TYPE)
            if checkpoint_model_type != args.model_type:
                raise RuntimeError(
                    f"Checkpoint model_type={checkpoint_model_type!r} does not match "
                    f"requested model_type={args.model_type!r}."
                )
            best_params = checkpoint["params"]
        except Exception as exc:
            raise RuntimeError(
                f"Could not load params from checkpoint {checkpoint_path!r}. "
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
            feature_set=args.feature_set,
            model_type=args.model_type,
        )

        print(f"\n{'=' * 70}")
        print("  BEST PARAMS")
        print("=" * 70)
        print(study.best_params)
        print(f"Best score: {study.best_value:.4f}")
        best_params = study.best_params

    models, score, metadata = train_xgboost_models_by_type(
        X_train,
        X_val,
        y_train,
        y_val,
        best_params,
        valid_target_indices,
        model_type=args.model_type,
    )
    print(f"Final score: {score:.4f}")

    calibration = None
    if args.calibrate_label_bias:
        calibration = calibrate_xgboost_label_bias(
            models,
            X_val,
            y_val,
            valid_target_indices,
            model_type=args.model_type,
            chain_order=metadata.get("chain_order"),
            k=args.calibration_k,
            alpha_max=args.calibration_alpha_max,
            steps=args.calibration_steps,
        )
        before = calibration["baseline_metrics"]
        after = calibration["calibrated_metrics"]
        print(
            "Calibration: "
            f"alpha={calibration['alpha']:.3f}, "
            f"score {before['score']:.4f} -> {after['score']:.4f}, "
            f"precision {before['precision']:.4f} -> {after['precision']:.4f}, "
            f"recall {before['recall']:.4f} -> {after['recall']:.4f}"
        )
        report = calibration["calibrated_report"]
        under_selected = sorted(report, key=lambda row: row["share_delta"])[:8]
        over_selected = sorted(report, key=lambda row: row["share_delta"], reverse=True)[:8]
        print("\nMost under-selected labels after calibration:")
        for row in under_selected:
            print(
                f"  {row['label']}: "
                f"true={row['true_share']:.3f}, selected={row['selected_share']:.3f}, "
                f"delta={row['share_delta']:.3f}, recall@{calibration['k']}={row['recall_at_k']:.3f}"
            )
        print("\nMost over-selected labels after calibration:")
        for row in over_selected:
            print(
                f"  {row['label']}: "
                f"true={row['true_share']:.3f}, selected={row['selected_share']:.3f}, "
                f"delta={row['share_delta']:.3f}, precision={row['precision_when_selected']:.3f}"
            )

    save_checkpoint(
        checkpoint_path,
        models,
        valid_target_indices,
        feature_columns,
        best_params,
        feature_set=args.feature_set,
        model_type=args.model_type,
        chain_order=metadata.get("chain_order"),
        calibration_bias=calibration["bias"] if calibration else None,
        calibration_metadata=calibration,
    )
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
