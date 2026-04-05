"""
Step 4: Clinical description generation.
Converts top-k important features + optional classification → Russian clinical text via Qwen2.5-14B.
Public API: generate_descriptions_batch(df, ...) → df with 'description' column.
"""

from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from generation.classification_types import ClassificationResult


# Global model cache
_model_cache = {"model": None, "tokenizer": None, "device": None}


def _parse_feature_entries(important_features: List[str]) -> List[Tuple[str, str]]:
    """Parse 'feature:value' strings into (russian_description, value) pairs."""
    from config.config import FEATURE_ROUTING

    results = []
    for entry in important_features:
        parts = entry.split(":", 1)
        feature_name = parts[0]
        label_value = parts[1] if len(parts) > 1 else ""

        routing = FEATURE_ROUTING.get(feature_name)
        if routing:
            _category_path, description, _unit = routing
        else:
            description = feature_name

        results.append((description, label_value))
    return results


def _build_prompt(
    parsed_features: List[Tuple[str, str]],
    classification: Optional[ClassificationResult],
) -> str:
    features_list = "\n".join(
        f"{i + 1}. {desc}: {value}"
        for i, (desc, value) in enumerate(parsed_features)
    )

    classification_block = ""
    if classification is not None:
        props_str = ", ".join(classification.properties) if classification.properties else "—"
        classification_block = f"""
КЛАССИФИКАЦИЯ ОБРАЗОВАНИЯ:
- Тип признаков: {classification.feature_type.value}
- Структура: {classification.structure.value}
- Свойства: {props_str}
- Итоговый класс: {classification.final_class}
"""

    return f"""Ты — медицинский ассистент-дерматолог. Составляй клинические описания дерматоскопических образований на русском языке. Описание должно быть профессиональным, лаконичным, 3–5 предложений. Используй только предоставленные данные. Не ставь диагноз.

Составь клиническое дерматоскопическое описание образования на основе следующих данных:

ВАЖНЕЙШИЕ ПРИЗНАКИ (топ-{len(parsed_features)}, по убыванию значимости):
{features_list}{classification_block}

Требования к описанию:
- 3–5 предложений
- Профессиональный клинический стиль
- Упоминай форму/симметрию, цвет/пигментацию, структуру/текстуру, контур
- Только русский язык

Описание:"""


def _load_model(
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
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


def _generate_text(prompt: str, model, tokenizer, device: torch.device, max_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7, top_p=0.9,
            repetition_penalty=1.2, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()


def _generate_single(
    important_features: List[str],
    classification: Optional[ClassificationResult],
    model, tokenizer, device: torch.device,
    max_tokens: int = 512,
) -> str:
    """Generate description for one set of features. Returns text or error string."""
    if not important_features:
        return "Недостаточно данных для описания."
    parsed = _parse_feature_entries(important_features)
    prompt = _build_prompt(parsed, classification)
    return _generate_text(prompt, model, tokenizer, device, max_tokens)


def _ensure_model(
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    device: Optional[torch.device] = None,
    use_cache: bool = True,
) -> Tuple[Any, Any, torch.device]:
    """Load model or return cached instance."""
    global _model_cache
    if use_cache and _model_cache["model"] is not None:
        return _model_cache["model"], _model_cache["tokenizer"], _model_cache["device"]

    model, tokenizer, device_used = _load_model(model_name, device)
    if use_cache:
        _model_cache["model"] = model
        _model_cache["tokenizer"] = tokenizer
        _model_cache["device"] = device_used
    return model, tokenizer, device_used


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_descriptions_batch(
    df: pd.DataFrame,
    classification_col: Optional[str] = "classification",
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    device: Optional[torch.device] = None,
    max_tokens: int = 512,
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
    has_classification = classification_col in df.columns

    try:
        model, tokenizer, device_used = _ensure_model(model_name, device, use_cache)

        descriptions = []
        iterator = tqdm(df.itertuples(index=False), total=len(df)) if verbose else df.itertuples(index=False)

        for row in iterator:
            row_dict = row._asdict()
            important_features = row_dict.get("important_labels", [])
            if not isinstance(important_features, list):
                important_features = []

            classification = row_dict.get(classification_col) if has_classification else None
            if classification is not None and not isinstance(classification, ClassificationResult):
                classification = None

            text = _generate_single(important_features, classification, model, tokenizer, device_used, max_tokens)
            descriptions.append(text)

        df["description"] = descriptions

    except ImportError:
        print("Warning: transformers module not installed.")
        df["description"] = "Модуль transformers не установлен."

    except Exception as e:
        print(f"Warning: Description generation failed: {e}")
        df["description"] = f"Ошибка генерации описания: {e}"

    return df
