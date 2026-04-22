"""Shared naming helpers for flat bucket columns."""

BUCKET_PREFIX = "bucket_"


def bucket_column_name(label_name: str) -> str:
    return f"{BUCKET_PREFIX}{label_name}"
