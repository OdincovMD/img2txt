"""
Batch API: Feature extraction for one or multiple images.
Input: DataFrame with 'image_path' column
Output: DataFrame with added feature columns
"""

from pathlib import Path
from typing import Dict, Optional, Union, Any
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from extraction.features import (
    extract_global_color_features_with_mask,
    extract_local_color_features_with_mask,
    extract_shape_features,
    extract_border_features,
    extract_texture_features,
)
from extraction.derived_features import merge_with_composite_features
from extraction.segmentation import main as segment_lesion


def images_to_df(
    image_dir: Union[str, Path],
) -> pd.DataFrame:
    """
    Scan a directory and return a DataFrame with 'image_path' column.
    Convenience function to build input for extract_features_batch.
    """
    image_dir = Path(image_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [p for p in image_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]
    return pd.DataFrame([{"image_path": str(p)} for p in sorted(paths)])


def _load_image_bgr(image_path: Union[str, Path]) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return image


def _extract_features_from_mask(image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    features = {}
    features.update(extract_global_color_features_with_mask(image, mask))
    features.update(extract_local_color_features_with_mask(image, mask))
    features.update(extract_shape_features(mask))
    features.update(extract_border_features(mask))
    features.update(extract_texture_features(image, mask))
    return merge_with_composite_features(features)


def _process_row(
    row_dict: Dict[str, Any],
    yolo_weights: Optional[str],
    unet_weights: Optional[str],
) -> Dict[str, Any]:
    image_path = Path(row_dict['image_path'])
    try:
        image = _load_image_bgr(image_path)
        mask = segment_lesion(str(image_path), yolo_weights, unet_weights)
        features_dict = _extract_features_from_mask(image, mask)
        return {"status": "success", **features_dict}

    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def extract_features_batch(
    df: pd.DataFrame,
    yolo_weights: Optional[str] = None,
    unet_weights: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract features for all rows in df.

    Args:
        df: DataFrame with 'image_path' column (absolute paths).
            May optionally contain 'image_id', 'filename', and other metadata columns —
            they are preserved unchanged.
        yolo_weights: Path to YOLO model weights (optional)
        unet_weights: Path to UNet fallback model weights (optional)
        verbose: Show progress bar

    Returns:
        Original df with added columns:
        - <individual features>: flattened feature values
        - status: 'success' or 'error'
        - error: error message if status == 'error'

    Example (single image):
        df = pd.DataFrame([{'image_path': '/data/lesion.jpg'}])
        df = extract_features_batch(df)

    Example (directory):
        df = images_to_df('/data/images/')
        df = extract_features_batch(df)
    """
    df = df.copy()

    iterator = df.itertuples(index=False)
    if verbose:
        iterator = tqdm(list(iterator), total=len(df))

    results = [_process_row(row._asdict(), yolo_weights, unet_weights) for row in iterator]
    results_df = pd.DataFrame(results, index=df.index)

    for col in results_df.columns:
        df[col] = results_df[col]

    return df
