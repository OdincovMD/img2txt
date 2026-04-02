"""
Batch API: Feature extraction for multiple images.
Input: image_dir + metadata_csv
Output: DataFrame with all features
"""

from pathlib import Path
from typing import Dict, Optional, Union, List, Any
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from core.features import (
    extract_global_color_features_with_mask,
    extract_local_color_features_with_mask,
    extract_shape_features,
    extract_border_features,
    extract_texture_features,
)
from core.segmentation import main as segment_lesion
from config.config import FEATURE_ROUTING, _safe_number
from data.data import MetaRow, build_row


def load_image_bgr(image_path: Union[str, Path]) -> np.ndarray:
    """Load image in BGR format (OpenCV convention)."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return image


def extract_features_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Extract all features (color, shape, border, texture) from image + mask.
    Returns dict with feature_name -> (value, description) tuples.
    """
    features = {}

    # Color features
    features.update(extract_global_color_features_with_mask(image, mask))
    features.update(extract_local_color_features_with_mask(image, mask))

    # Shape, border, texture
    features.update(extract_shape_features(mask))
    features.update(extract_border_features(mask))
    features.update(extract_texture_features(image, mask))

    return features


def extract_single_image_features(
    image_path: Union[str, Path],
    meta: MetaRow,
    yolo_weights: Optional[str] = None,
    unet_weights: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract features from a single image.
    Returns a dataset row dict with metadata + features_json.
    """
    image_path = Path(image_path)

    try:
        # Load image
        image = load_image_bgr(image_path)

        # Segment lesion
        mask = segment_lesion(str(image_path), yolo_weights, unet_weights)

        # Extract features
        features_dict = extract_features_from_mask(image, mask)

        # Build dataset row
        row = build_row(meta, features_dict)
        row['status'] = 'success'
        return row

    except Exception as e:
        return {
            'image_id': meta.image_id,
            'filename': meta.filename,
            'class_dir': meta.class_dir,
            'status': 'error',
            'error': str(e),
        }


def extract_features_batch(
    image_dir: Union[str, Path],
    metadata_csv: Union[str, Path],
    yolo_weights: Optional[str] = None,
    unet_weights: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract features from all images in a directory.

    Args:
        image_dir: Directory containing images
        metadata_csv: CSV with columns: image_id, filename, class_dir, [diagnosis, feature_type, structure, properties]
        yolo_weights: Path to YOLO model weights (optional)
        unet_weights: Path to UNet fallback model weights (optional)
        verbose: Show progress bar

    Returns:
        DataFrame with columns:
        - image_id, filename, class_dir, diagnosis, feature_type, structure, properties (from CSV)
        - features_json: structured dict of all extracted features
        - <individual features>: flattened feature columns
        - status: 'success' or 'error'
        - error: error message if status == 'error'
    """
    image_dir = Path(image_dir)
    metadata_csv = Path(metadata_csv)

    # Load metadata
    df_meta = pd.read_csv(metadata_csv)

    rows = []

    # Process each image
    iterator = tqdm(df_meta.itertuples(index=False), total=len(df_meta)) if verbose else df_meta.itertuples(index=False)

    for row_tuple in iterator:
        row_dict = row_tuple._asdict()

        # Build MetaRow
        meta = MetaRow(
            image_id=row_dict.get('image_id'),
            filename=row_dict.get('filename'),
            class_dir=row_dict.get('class_dir'),
            diagnosis=row_dict.get('diagnosis'),
            feature_type=row_dict.get('feature_type'),
            structure=row_dict.get('structure'),
            properties=row_dict.get('properties'),
        )

        # Build full image path
        image_path = image_dir / meta.filename

        # Extract features
        result_row = extract_single_image_features(
            image_path,
            meta,
            yolo_weights=yolo_weights,
            unet_weights=unet_weights,
        )

        rows.append(result_row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Add individual feature columns (flatten features_json)
    if 'features_json' in df.columns:
        def extract_feature_value(features_json, feature_key):
            if features_json is None:
                return None
            if isinstance(features_json, dict):
                for cat_name, cat_dict in features_json.items():
                    if isinstance(cat_dict, dict):
                        if feature_key in cat_dict:
                            val = cat_dict[feature_key]
                            return _safe_number(val)
            return None

        # Add columns for all known features
        for feature_key in FEATURE_ROUTING.keys():
            df[feature_key] = df['features_json'].apply(
                lambda x: extract_feature_value(x, feature_key)
            )

    return df
