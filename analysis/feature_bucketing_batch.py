"""
Batch API: Feature bucketing for entire datasets.
Converts numeric features -> categorical labels/buckets for all images
Input: DataFrame with features (from feature_extraction_batch)
Output: DataFrame with added labels and features_json columns
"""

from typing import Dict, Any, Optional
import json
import pandas as pd
from tqdm import tqdm

from analysis.threshold_rules import features_to_labels
from config.config import FEATURE_ROUTING
from config.importance_config import ACTIVE_LABELS


def bucket_features_batch(
    df: pd.DataFrame,
    features_col: str = 'features_json',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convert numeric features to categorical labels for all rows in DataFrame.

    Args:
        df: DataFrame with feature columns (numeric values)
        features_col: Name of column containing features_json (if exists)
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
        features_dict = _extract_numeric_features(row_dict)

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


def _extract_numeric_features(row_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract numeric feature values from a row dict.
    Handles:
    - Direct numeric columns
    - features_json (dict of dicts)
    - features_dict
    """
    features = {}

    # First, try to extract from features_json (structured format)
    if 'features_json' in row_dict:
        features_json = row_dict['features_json']
        if isinstance(features_json, str):
            try:
                features_json = json.loads(features_json)
            except:
                features_json = {}
        if isinstance(features_json, dict):
            features.update(_flatten_features_json(features_json))

    # Then, extract numeric columns directly (overwrites if duplicates)
    for col_name, value in row_dict.items():
        if col_name in FEATURE_ROUTING:
            if isinstance(value, (int, float)) and not pd.isna(value):
                features[col_name] = float(value)

    return features


def _flatten_features_json(features_json: Dict[str, Any]) -> Dict[str, float]:
    """
    Flatten nested features_json structure into flat dict of numeric values.
    """
    flat = {}

    def flatten_dict(d, parent_key=''):
        for k, v in d.items():
            if isinstance(v, dict):
                flatten_dict(v, parent_key)
            elif isinstance(v, (int, float)) and not pd.isna(v):
                flat[k] = float(v)
            elif isinstance(v, (list, tuple)):
                # Skip lists/tuples (like dominant_colors)
                pass

    flatten_dict(features_json)
    return flat


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


def get_label_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get statistics on labels across the dataset.
    Returns DataFrame with label value counts and percentages.
    """
    stats_list = []

    # Iterate through label columns
    for col in df.columns:
        if col.startswith('labels'):
            continue

        if 'labels' in col or col == 'labels_json':
            continue

        # Try to extract label values
        if col in df.columns and df[col].dtype == 'object':
            try:
                value_counts = df[col].value_counts()
                for value, count in value_counts.items():
                    stats_list.append({
                        'label': col,
                        'value': value,
                        'count': count,
                        'percentage': (count / len(df)) * 100,
                    })
            except:
                pass

    return pd.DataFrame(stats_list) if stats_list else pd.DataFrame()
