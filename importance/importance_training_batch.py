"""
Training API: Feature importance model training and pseudo-label generation.
For development/training only. Not used in local inference runs.

Two-stage workflow:
1. Generate pseudo-labels: add_pseudo_important_labels() — artificial markup without expert input
2. Train model: train_importance() — trains neural network from pseudo-labeled dataset
"""

from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

from importance.pseudo_importance import add_pseudo_important_labels
from importance.train_importance import train_importance


def generate_pseudo_labels(
    df: pd.DataFrame,
    labels_col: str = "labels",
    features_col: str = "features_json",
    diagnosis_col: str = "diagnosis",
    structure_col: str = "structure",
    properties_col: str = "properties",
    out_col: str = "pseudo_important_labels",
    top_k: int = 10,
    mode: str = "combined",
    z_weight: float = 1.0,
    prior_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Generate pseudo-labels (artificial markup) without expert input.

    Two methods combined:
    - Statistical: Robust z-score of numeric features across dataset
    - Metadata-based: Weights from diagnosis/structure/properties text

    Args:
        df: DataFrame with features and metadata from Steps 1-2
        labels_col: Column name with diagnostic labels dict
        features_col: Column name with features_json
        diagnosis_col: Column name with diagnosis text
        structure_col: Column name with structure text
        properties_col: Column name with properties text
        out_col: Output column name for pseudo-labels
        top_k: Number of top features to select
        mode: "statistical" (z-score only) or "combined" (z-score + metadata)
        z_weight: Weight for z-score component (statistical method)
        prior_weight: Weight for metadata prior component

    Returns:
        DataFrame with added out_col: list of top-k feature strings ("key:value")
    """
    return add_pseudo_important_labels(
        df,
        labels_step1_col=labels_col,
        features_col=features_col,
        diagnosis_col=diagnosis_col,
        structure_col=structure_col,
        properties_col=properties_col,
        out_col=out_col,
        top_k=top_k,
        mode=mode,
        z_weight=z_weight,
        prior_weight=prior_weight,
    )


def train_importance_model(
    data_csv: Union[str, Path],
    image_dir: Union[str, Path],
    *,
    mask_dir: Optional[Union[str, Path]] = None,
    image_col: str = "filename",
    image_id_col: str = "image_id",
    val_ratio: float = 0.15,
    splits_file: Optional[Union[str, Path]] = None,
    backbone: Literal["efficientnet_b0", "resnet18", "resnet34"] = "efficientnet_b0",
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-4,
    image_size: int = 224,
    use_mask: bool = False,
    out_dir: Union[str, Path] = "importance_checkpoints",
    seed: int = 42,
) -> float:
    """
    Train importance ranking model from pseudo-labeled dataset.

    Requirements:
    - CSV must contain: image_id, filename (or custom columns), important_labels

    Args:
        data_csv: Path to CSV with metadata + images + important_labels
        image_dir: Directory containing images
        mask_dir: Optional directory with mask files
        image_col: Column name with image filenames
        image_id_col: Column name with image IDs
        val_ratio: Fraction of data for validation (0.0-1.0)
        splits_file: Optional JSON file with predefined train/val splits
        backbone: Model backbone ("efficientnet_b0", "resnet18", "resnet34")
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        image_size: Input image size for model
        use_mask: If True, model expects 4-channel input (RGB + mask)
        out_dir: Directory to save checkpoints (best.pt, splits.json)
        seed: Random seed for reproducibility

    Returns:
        Best validation score ((P@10 + R@10) / 2)
    """
    best_score = train_importance(
        data_csv=data_csv,
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_col=image_col,
        image_id_col=image_id_col,
        val_ratio=val_ratio,
        splits_file=splits_file,
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        image_size=image_size,
        use_mask=use_mask,
        out_dir=out_dir,
        seed=seed,
    )

    return best_score


def prepare_for_training(
    df: pd.DataFrame,
    csv_path: Union[str, Path],
    labels_col: str = "labels",
    features_col: str = "features_json",
    diagnosis_col: str = "diagnosis",
    structure_col: str = "structure",
    properties_col: str = "properties",
    top_k: int = 10,
    z_weight: float = 1.5,
    prior_weight: float = 0.8,
) -> str:
    """
    Convenience function: Generate pseudo-labels and save CSV for training.

    Args:
        df: DataFrame with features and metadata from Steps 1-2
        csv_path: Path to save prepared CSV
        labels_col: Column name with diagnostic labels
        features_col: Column name with features_json
        diagnosis_col: Column name with diagnosis text
        structure_col: Column name with structure text
        properties_col: Column name with properties text
        top_k: Number of top features for pseudo-labels
        z_weight: Weight for statistical method
        prior_weight: Weight for metadata prior

    Returns:
        Path to saved CSV (str)
    """
    df_prepared = generate_pseudo_labels(
        df,
        labels_col=labels_col,
        features_col=features_col,
        diagnosis_col=diagnosis_col,
        structure_col=structure_col,
        properties_col=properties_col,
        out_col="important_labels",
        top_k=top_k,
        mode="combined",
        z_weight=z_weight,
        prior_weight=prior_weight,
    )

    csv_path = Path(csv_path)
    df_prepared.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path.resolve()} ({len(df_prepared)} rows)")

    return str(csv_path)
