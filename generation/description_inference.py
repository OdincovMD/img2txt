"""
Step 4: Clinical description generation.
Converts top-k important features + optional classification → Russian clinical text via Qwen2.5-7B.
Public API: generate_descriptions_batch(df, ...) → df with 'description' column.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import pandas as pd
import torch
from tqdm import tqdm

from generation.classification_types import ClassificationResult
from generation.description_templates import build_messages, parse_feature_entries


@dataclass
class ModelCacheEntry:
    model_name: str
    model: Any
    tokenizer: Any
    device: torch.device


_model_cache: Optional[ModelCacheEntry] = None


def reset_description_model_cache() -> None:
    """Drop cached LLM resources so the next call reloads them."""
    global _model_cache
    _model_cache = None


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


def _load_model(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: Optional[torch.device] = None,
) -> Tuple[Any, Any, torch.device]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

    model.eval()
    print(f"Model loaded on {device}")
    return model, tokenizer, device


def _generate_text(
    messages: List[Dict[str, str]], model, tokenizer, device: torch.device, max_tokens: int = 300,
) -> str:
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3, top_p=0.85,
            repetition_penalty=1.15, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def _generate_single(
    important_features: List[str],
    classification: Optional[ClassificationResult],
    model, tokenizer, device: torch.device,
    max_tokens: int = 256,
) -> str:
    """Generate description for one set of features. Returns text or error string."""
    if not important_features:
        return "Недостаточно данных для описания."
    grouped = parse_feature_entries(important_features)
    messages = build_messages(grouped, classification)
    return _generate_text(messages, model, tokenizer, device, max_tokens)


def _ensure_model(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: Optional[torch.device] = None,
    use_cache: bool = True,
) -> Tuple[Any, Any, torch.device]:
    """Load model or return cached instance."""
    global _model_cache
    requested_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cached = _model_cache
    if use_cache and cached is not None:
        if cached.model_name == model_name and cached.device.type == requested_device.type:
            return cached.model, cached.tokenizer, cached.device

    model, tokenizer, device_used = _load_model(model_name, requested_device)
    if use_cache:
        _model_cache = ModelCacheEntry(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            device=device_used,
        )
    return model, tokenizer, device_used


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
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: Optional[torch.device] = None,
    max_tokens: int = 300,
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
        model_name: HuggingFace model ID
        device: torch device (defaults to cuda if available)
        max_tokens: Max tokens per description
        use_cache: Cache model in memory for subsequent calls
        verbose: Show progress bar

    Returns:
        df with added column:
        - description: Russian clinical text (3-5 sentences)
    """
    df = df.copy()
    has_classification = classification_col is not None and classification_col in df.columns

    try:
        model, tokenizer, device_used = _ensure_model(model_name, device, use_cache)

        descriptions = []
        iterator = tqdm(df.itertuples(index=False), total=len(df)) if verbose else df.itertuples(index=False)

        for row in iterator:
            important_features, classification = _extract_generation_inputs(
                row,
                classification_col,
                has_classification,
            )
            text = _generate_single(important_features, classification, model, tokenizer, device_used, max_tokens)
            descriptions.append(text)

        df["description"] = descriptions

    except ImportError:
        print("Warning: transformers module not installed.")
        df["description"] = ["Модуль transformers не установлен." for _ in range(len(df))]

    except Exception as e:
        print(f"Warning: Description generation failed: {e}")
        df["description"] = [f"Ошибка генерации описания: {e}" for _ in range(len(df))]

    return cast(pd.DataFrame, df)
