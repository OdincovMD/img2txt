"""
Batch API: Feature bucketing for entire datasets.
Converts numeric features -> categorical labels/buckets for all images
Input: DataFrame with features (from feature_extraction_batch)
Output: DataFrame with added labels and features_json columns
"""

from typing import Dict, Any
import json
import pandas as pd
from tqdm import tqdm

from analysis.threshold_rules import features_to_labels
from config.config import FEATURE_ROUTING
from config.importance_config import ACTIVE_LABELS


def bucket_features_batch(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convert numeric features to categorical labels for all rows in DataFrame.

    Args:
        df: DataFrame with feature columns (numeric values)
        verbose: Show progress bar

    Returns:
        DataFrame with added columns:
        - features_organized: Hierarchical structure of features
        - labels: Dict of diagnostic labels for each image
        - labels_json: JSON string of labels
    """
    df = df.copy()

    results_list = []

    iterator = tqdm(df.itertuples(index=False), total=len(df)) if verbose else df.itertuples(index=False)

    for row_tuple in iterator:
        row_dict = row_tuple._asdict()

        # Extract numeric features from row
        features_dict = _extract_features(row_dict)

        # Apply threshold rules (вычисляются все признаки)
        all_labels = features_to_labels(features_dict)

        # Оставляем только признаки, прошедшие порог частоты
        labels = {k: v for k, v in all_labels.items() if k in ACTIVE_LABELS}

        # Organize features into hierarchical structure
        features_organized = _organize_features_structure(features_dict)

        results_list.append({
            'features_dict': features_dict,
            'features_organized': features_organized,
            'labels': labels,
            'labels_json': json.dumps(labels, ensure_ascii=False),
        })

    # Convert results to DataFrame and add to original
    results_df = pd.DataFrame(results_list, index=df.index)
    for col in results_df.columns:
        df[col] = results_df[col]

    return df


def _extract_features(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract feature values from a row dict (direct columns from Step 1)."""
    features = {}
    for key in FEATURE_ROUTING:
        value = row_dict.get(key)
        if isinstance(value, (int, float)) and not pd.isna(value):
            features[key] = float(value)
        elif isinstance(value, (list, tuple)):
            # dominant_colors_lesion и подобные нечисловые признаки
            features[key] = value
    return features


def _organize_features_structure(features_dict: Dict[str, float]) -> Dict[str, Dict]:
    """
    Organize flat feature dict into hierarchical structure by category.
    Uses FEATURE_ROUTING to determine category for each feature.
    """
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


