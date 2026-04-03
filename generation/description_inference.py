"""
Inference API: Clinical description generation for a single dermoscopy finding.
Converts top-10 important features + classification → Russian clinical text

Pipeline: important_features + ClassificationResult → Mistral-7B (local) → clinical description
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
from generation.classification_types import ClassificationResult


# Global model cache for the notebook session
_model_cache = {"model": None, "tokenizer": None, "device": None}


def _parse_feature_entries(
    important_features: List[str],
) -> List[Tuple[str, str]]:
    """
    Parse top-10 feature entries into (russian_description, label_value) pairs.

    Each entry has format: "feature_name:label_value"
    Some label values are compound: "вариабельность_оттенка:высокая"

    We look up FEATURE_ROUTING[feature_name] to get a Russian description.
    If feature not in FEATURE_ROUTING, fall back to the raw feature name.

    Returns:
        List of (russian_description, label_value) tuples
    """
    from config.config import FEATURE_ROUTING

    results = []
    for entry in important_features:
        # Split on first colon only: "feature_name:rest"
        parts = entry.split(":", 1)
        feature_name = parts[0]
        label_value = parts[1] if len(parts) > 1 else ""

        # Look up Russian description from FEATURE_ROUTING
        routing = FEATURE_ROUTING.get(feature_name)
        if routing:
            # routing = (category_path, description, unit)
            _category_path, description, _unit = routing
        else:
            # Fallback: use raw feature name
            description = feature_name

        results.append((description, label_value))

    return results


def _build_prompt(
    parsed_features: List[Tuple[str, str]],
    classification: Optional[ClassificationResult],
) -> str:
    """
    Build user message for Mistral.

    Args:
        parsed_features: List of (russian_description, label_value) from _parse_feature_entries
        classification: Optional ClassificationResult

    Returns:
        Full prompt string (system prompt included in-line for Mistral format)
    """
    # Build features list
    features_list = "\n".join(
        f"{i + 1}. {desc}: {value}"
        for i, (desc, value) in enumerate(parsed_features)
    )

    # Build classification block (optional)
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

    prompt = f"""Ты — медицинский ассистент-дерматолог. Составляй клинические описания дерматоскопических образований на русском языке. Описание должно быть профессиональным, лаконичным, 3–5 предложений. Используй только предоставленные данные. Не ставь диагноз.

Составь клиническое дерматоскопическое описание образования на основе следующих данных:

ВАЖНЕЙШИЕ ПРИЗНАКИ (топ-{len(parsed_features)}, по убыванию значимости):
{features_list}{classification_block}

Требования к описанию:
- 3–5 предложений
- Профессиональный клинический стиль
- Упоминай форму/симметрию, цвет/пигментацию, структуру/текстуру, контур
- Только русский язык

Описание:"""

    return prompt


def _load_mistral_model(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    device: Optional[torch.device] = None,
) -> Tuple[Any, Any, torch.device]:
    """
    Load Mistral-7B model and tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model ID
        device: torch device (cuda/cpu). Defaults to cuda if available.

    Returns:
        (model, tokenizer, device) tuple
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {model_name} on {device}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with appropriate precision
    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
        )

    model.eval()
    print(f"Model loaded on {device}")

    return model, tokenizer, device


def _generate_with_mistral(
    prompt: str,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    max_tokens: int = 512,
) -> str:
    """
    Generate text using Mistral model.

    Args:
        prompt: Input prompt
        model: Mistral model instance
        tokenizer: Tokenizer instance
        device: torch device
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated part (remove prompt)
    response = generated_text[len(prompt):].strip()

    return response


def generate_description(
    important_features: List[str],
    classification: Optional[ClassificationResult] = None,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    device: Optional[torch.device] = None,
    max_tokens: int = 512,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Generate a Russian clinical description for a dermoscopy finding using Mistral-7B.

    This is the primary public API for Step 4 (text generation) of the pipeline.

    Args:
        important_features: Top-10 ranked features from rank_features(),
            format ["feature_name:label_value", ...].
        classification: Optional ClassificationResult with dermoscopy classification.
            If None, the classification block is omitted from the prompt.
        model_name: HuggingFace model ID for Mistral. Defaults to Mistral-7B-Instruct-v0.1.
        device: torch device (cuda/cpu). Defaults to cuda if available.
        max_tokens: Max tokens for generated description. Defaults to 512.
        use_cache: Cache model in memory for subsequent calls. Defaults to True.

    Returns:
        {
            'description': str,         # Russian clinical text, 3-5 sentences
            'model': str,               # Model used
            'has_classification': bool, # Whether classification was provided
            'feature_count': int,       # Number of features used
        }

    Notes:
        - This function never raises exceptions. Errors are caught and reported
          in the 'description' field with a warning printed to stdout.
        - First call loads the model (may take 30-60 seconds on CPU, faster on GPU).
        - Subsequent calls use cached model if use_cache=True.
        - Empty important_features → description: "Недостаточно данных для описания."
        - Missing transformers module → description: "Модуль transformers не установлен."
        - Model loading error → description: "Ошибка загрузки модели: {error}."
    """
    global _model_cache

    # Edge case: empty features
    if not important_features:
        return {
            "description": "Недостаточно данных для описания.",
            "model": model_name,
            "has_classification": classification is not None,
            "feature_count": 0,
        }

    try:
        # Load or use cached model
        if use_cache and _model_cache["model"] is not None:
            model, tokenizer, device_used = (
                _model_cache["model"],
                _model_cache["tokenizer"],
                _model_cache["device"],
            )
        else:
            model, tokenizer, device_used = _load_mistral_model(model_name, device)
            if use_cache:
                _model_cache["model"] = model
                _model_cache["tokenizer"] = tokenizer
                _model_cache["device"] = device_used

        # Parse features
        parsed = _parse_feature_entries(important_features)

        # Build prompt (Mistral format)
        prompt = _build_prompt(parsed, classification)

        # Generate description
        description = _generate_with_mistral(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device_used,
            max_tokens=max_tokens,
        )

        return {
            "description": description,
            "model": model_name,
            "has_classification": classification is not None,
            "feature_count": len(important_features),
        }

    except ImportError:
        error_msg = "Модуль transformers не установлен."
        print(f"Warning: {error_msg}")
        return {
            "description": error_msg,
            "model": model_name,
            "has_classification": classification is not None,
            "feature_count": len(important_features),
        }

    except Exception as e:
        error_msg = f"Ошибка генерации описания: {str(e)}."
        print(f"Warning: {error_msg}")
        return {
            "description": error_msg,
            "model": model_name,
            "has_classification": classification is not None,
            "feature_count": len(important_features),
        }
