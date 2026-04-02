"""
Inference API: Feature importance ranking for a single image or batch.
Converts features + labels from Step 2 into ranked top-10 important features.
Input: features_dict + labels (from feature_bucketing)
Output: List of top-10 features ranked by importance (model-based or pseudo)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd
import torch

from importance.importance_inference import load_model, predict_top10, add_important_labels_to_dataframe


def rank_features(
    features_dict: Dict[str, float],
    labels: Dict[str, Any],
    image: Optional[Any] = None,
    importance_model_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    image_size: int = 224,
    k: int = 10,
    mask: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Rank features by importance for a single image using trained model.

    Args:
        features_dict: Dict of numeric features (feature_name -> value)
        labels: Dict of diagnostic labels (from bucket_features)
        image: Optional image array (H,W,3) for model inference
        importance_model_path: Path to importance model checkpoint
        device: torch device for inference
        image_size: Image size for model input
        k: Number of top features to return
        mask: Optional mask array for 4-channel inference

    Returns:
        {
            'important_features': list of top-k feature labels (["feature_name:value", ...]),
            'has_model': bool (True if model was provided and used),
        }
    """
    result = {
        'important_features': [],
        'has_model': False,
    }

    # Only rank with model if both image and model path provided
    if image is not None and importance_model_path is not None:
        try:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model = load_model(importance_model_path, device=device)
            result['has_model'] = True

            top_k_labels = predict_top10(
                model=model,
                image=image,
                labels_dict=labels,
                device=device,
                image_size=image_size,
                k=k,
                mask=mask,
            )
            result['important_features'] = top_k_labels

        except Exception as e:
            print(f"Warning: Could not rank features with model: {e}")
            result['has_model'] = False

    return result


def rank_features_batch(
    df: pd.DataFrame,
    importance_model_path: Optional[Union[str, Path]] = None,
    image_dir: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    image_size: int = 224,
    k: int = 10,
    mask_dir: Optional[Union[str, Path]] = None,
    image_col: str = "filename",
    image_id_col: str = "image_id",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Rank features by importance for entire dataset using trained model.

    Args:
        df: DataFrame with features and labels from Step 2
        importance_model_path: Path to importance model checkpoint
        image_dir: Directory containing images (required for model-based ranking)
        device: torch device for inference
        image_size: Image size for model input
        k: Number of top features per image
        mask_dir: Optional directory with mask files
        image_col: Column name with image filenames
        image_id_col: Column name with image IDs
        verbose: Show progress bar

    Returns:
        DataFrame with added column: important_labels (list of top-k feature labels)
    """
    df = df.copy()

    if importance_model_path is not None and image_dir is not None:
        try:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model = load_model(importance_model_path, device=device)

            df = add_important_labels_to_dataframe(
                df,
                model=model,
                device=device,
                image_dir=image_dir,
                image_col=image_col,
                image_id_col=image_id_col,
                mask_dir=mask_dir,
                image_size=image_size,
                k=k,
            )

        except Exception as e:
            print(f"Warning: Could not rank features: {e}")
            df['important_labels'] = [[] for _ in range(len(df))]

    else:
        # No model provided: create empty important_labels column
        df['important_labels'] = [[] for _ in range(len(df))]

    return df
