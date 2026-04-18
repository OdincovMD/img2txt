"""
Batch API for pipeline bucketing.
Input: DataFrame with `image_path` and numeric features.
Output: same DataFrame + `bucket_*` columns.
"""

import pandas as pd
from tqdm import tqdm

from bucketing.threshold_rules import (
    bucket_columns_from_labels,
    bucket_feature_values,
    extract_raw_features,
)


def bucket_features_batch(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convert raw numeric features to flat bucket columns for each row.

    Args:
        df: DataFrame with raw feature columns.
        verbose: Show progress bar.

    Returns:
        Copy of input df with added `bucket_*` categorical columns.
    """
    result_df = df.copy()

    iterator = result_df.itertuples(index=False)
    if verbose:
        iterator = tqdm(list(iterator), total=len(result_df))

    bucket_rows = []
    for row in iterator:
        raw_features = extract_raw_features(row._asdict())
        label_values = bucket_feature_values(raw_features)
        bucket_rows.append(bucket_columns_from_labels(label_values))

    bucket_df = pd.DataFrame(bucket_rows, index=result_df.index)
    for col in bucket_df.columns:
        result_df[col] = bucket_df[col]

    return result_df
