"""
Step 4: Clinical description generation.
Converts top-k important features + optional classification → Russian clinical text via Groq.
Public API: generate_descriptions_batch(df, ...) → df with 'description' column.
"""

from typing import Any, List, Optional, Tuple, cast

import pandas as pd
from tqdm import tqdm

from generation.classification_types import ClassificationResult
from generation.llm_client import generate_description
from generation.description_templates import build_messages, parse_feature_entries


def reset_description_model_cache() -> None:
    """Compatibility no-op; Groq client resources are managed in llm_client."""
    return None


def _normalize_important_features(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, tuple):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        from model.model import parse_expert_labels

        return parse_expert_labels(raw)
    return []


def _normalize_classification(raw: Any) -> Optional[ClassificationResult]:
    if isinstance(raw, ClassificationResult):
        return raw
    return None


def _generate_single(
    important_features: List[str],
    classification: Optional[ClassificationResult],
    model_name: str,
    max_tokens: int = 256,
) -> str:
    """Generate description for one set of features. Returns text or error string."""
    if not important_features:
        return "Недостаточно данных для описания."
    grouped = parse_feature_entries(important_features)
    messages = build_messages(grouped, classification)
    return generate_description(messages, model=model_name, max_tokens=max_tokens)


def _extract_generation_inputs(
    row: Any,
    classification_col: Optional[str],
    has_classification: bool,
) -> Tuple[List[str], Optional[ClassificationResult]]:
    row_dict = row._asdict()
    important_features = _normalize_important_features(row_dict.get("important_labels"))
    classification = _normalize_classification(row_dict.get(classification_col)) if has_classification else None
    return important_features, classification


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_descriptions_batch(
    df: pd.DataFrame,
    classification_col: Optional[str] = "classification",
    model_name: str = "openai/gpt-oss-120b",
    device: Optional[Any] = None,
    max_tokens: int = 800,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Step 4: Generate Russian clinical descriptions for all rows in df.

    Args:
        df: DataFrame with 'important_labels' column (from step 3).
            Optional: column with ClassificationResult objects (see classification_col).
        classification_col: Column name containing ClassificationResult objects.
            If column doesn't exist or value is None, classification block is omitted.
        model_name: Groq model ID
        device: Deprecated compatibility argument, ignored.
        max_tokens: Max tokens per description
        use_cache: Deprecated compatibility argument, ignored.
        verbose: Show progress bar

    Returns:
        df with added column:
        - description: Russian clinical text (3-5 sentences)
    """
    df = df.copy()
    has_classification = classification_col is not None and classification_col in df.columns

    try:
        descriptions = []
        iterator = tqdm(df.itertuples(index=False), total=len(df)) if verbose else df.itertuples(index=False)

        for row in iterator:
            important_features, classification = _extract_generation_inputs(
                row,
                classification_col,
                has_classification,
            )
            text = _generate_single(important_features, classification, model_name, max_tokens)
            descriptions.append(text)

        df["description"] = descriptions

    except Exception as e:
        print(f"Warning: Description generation failed: {e}")
        df["description"] = [f"Ошибка генерации описания: {e}" for _ in range(len(df))]

    return cast(pd.DataFrame, df)
