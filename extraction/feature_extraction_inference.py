"""
Inference API: Feature extraction for a single image.
Pipeline: feature extraction → feature bucketing → importance ranking (optional)
Input: image_path (+ optional segmentation models and importance model)
Output: {features, features_dict, labels, important_features, image_size, mask_shape}
"""

from pathlib import Path
from typing import Dict, Optional, Union, Any
import numpy as np
import cv2
from PIL import Image

from core.features import (
    extract_global_color_features_with_mask,
    extract_local_color_features_with_mask,
    extract_shape_features,
    extract_border_features,
    extract_texture_features,
)
from core.segmentation import main as segment_lesion
from analysis.feature_bucketing_inference import bucket_features
from importance.importance_inference import load_model, predict_top10
import torch


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


def extract_features_and_rank(
    image_path: Union[str, Path],
    yolo_weights: Optional[str] = None,
    unet_weights: Optional[str] = None,
    importance_model_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Extract features from a single image, bucket them, and optionally rank by importance.

    Pipeline:
    1. Extract features (numeric values)
    2. Bucket features (convert to categorical labels)
    3. Rank top-10 important features (if importance model provided)

    Args:
        image_path: Path to the lesion image
        yolo_weights: Path to YOLO model weights (optional)
        unet_weights: Path to UNet fallback model weights (optional)
        importance_model_path: Path to importance ranking model checkpoint (optional)
        device: torch device for inference (defaults to cuda/cpu)

    Returns:
        {
            'features': dict of all extracted features (feature_name -> (value, description)),
            'features_dict': dict of just numeric values (feature_name -> value),
            'labels': dict of bucketed categorical labels (feature_name -> label),
            'important_features': list of top-10 feature labels if model provided,
            'image_size': (height, width),
            'mask_shape': mask dimensions,
        }
    """
    # Load image
    image = load_image_bgr(image_path)
    h, w = image.shape[:2]

    # Segment lesion
    mask = segment_lesion(str(image_path), yolo_weights, unet_weights)

    # Extract features
    features_with_desc = extract_features_from_mask(image, mask)

    # Extract numeric values for ranking
    features_dict = {k: v[0] if isinstance(v, tuple) else v
                    for k, v in features_with_desc.items()}

    # Bucket features to get categorical labels
    bucketed = bucket_features(features_dict)
    labels = bucketed['labels']

    result = {
        'features': features_with_desc,
        'features_dict': features_dict,
        'labels': labels,
        'image_size': (h, w),
        'mask_shape': mask.shape,
    }

    # Rank features by importance if model provided
    if importance_model_path is not None:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load importance model
        try:
            model = load_model(importance_model_path, device=device)
        except Exception as e:
            print(f"Warning: Could not load importance model: {e}")
            model = None

        if model is not None:
            # Prepare image for model
            # Check if model expects 4 channels (RGB+mask)
            try:
                in_channels = next(model.backbone.parameters()).shape[1]
            except:
                in_channels = 3

            # Prepare input
            if in_channels == 4:
                # Include mask as 4th channel
                image_input = image.copy()
                mask_normalized = (mask.astype(float) / 255.0 * 255).astype(np.uint8)
                image_input = np.dstack([image, mask_normalized])
            else:
                image_input = image

            # Get top-10 features from bucketed labels
            try:
                top_10_labels = predict_top10(
                    model=model,
                    image=image_input,
                    labels_dict=labels,
                    device=device,
                    image_size=224,
                    k=10,
                    mask=mask if in_channels == 4 else None,
                )
                result['important_features'] = top_10_labels
            except Exception as e:
                print(f"Warning: Could not rank features: {e}")
                result['important_features'] = []

    return result
