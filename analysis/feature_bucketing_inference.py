"""
Inference API: Feature bucketing for a single image.
Converts numeric features -> categorical labels/buckets
Input: features dict (from feature_extraction_inference)
Output: {features_json, labels} - organized bucketed data
"""

from typing import Dict, Any, Optional
import json


def bucket_features(
    features_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert numeric features into categorical labels using threshold rules.

    Args:
        features_dict: Dict of numeric features (feature_name -> value)

    Returns:
        {
            'features_json': structured feature dict (category -> {feature -> value}),
            'labels': dict of diagnostic labels (label_name -> value),
            'raw_features': original features_dict,
        }
    """
    from analysis.threshold_rules import features_to_labels

    # Organize features into structure (for reference)
    features_json = _organize_features_structure(features_dict)

    # Apply threshold rules to get labels
    labels = features_to_labels(features_dict)

    return {
        'features_json': features_json,
        'labels': labels,
        'raw_features': features_dict,
    }


def _organize_features_structure(features_dict: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Organize flat feature dict into hierarchical structure by category.
    Uses FEATURE_ROUTING to determine category for each feature.
    """
    from config.config import FEATURE_ROUTING

    structure = {
        'color': {'local': {}, 'global': {}},
        'shape': {},
        'border': {},
        'texture': {},
    }

    for feature_name, value in features_dict.items():
        if feature_name not in FEATURE_ROUTING:
            continue

        category_path, description, unit = FEATURE_ROUTING[feature_name]
        parts = category_path.split('.')

        # Navigate to correct location in structure
        current = structure
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Add feature
        final_key = parts[-1]
        if final_key not in current:
            current[final_key] = {}
        current[final_key][feature_name] = value

    return structure


def format_labels_for_display(labels: Dict[str, Any]) -> str:
    """
    Format labels dict into readable string for display/reporting.
    """
    if not labels:
        return "No labels"

    lines = []
    for category, value in sorted(labels.items()):
        if isinstance(value, dict):
            lines.append(f"{category}:")
            for k, v in sorted(value.items()):
                lines.append(f"  {k}: {v}")
        elif isinstance(value, (list, tuple)):
            lines.append(f"{category}: {', '.join(map(str, value))}")
        else:
            lines.append(f"{category}: {value}")

    return "\n".join(lines)
