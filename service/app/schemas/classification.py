"""Helpers for classification payloads used in description generation."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def normalize_classification_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    properties = payload.get("properties") or []
    return {
        "feature_type": str(payload.get("feature_type") or "").strip(),
        "structure": str(payload.get("structure") or "").strip(),
        "properties": [str(item).strip() for item in properties if str(item).strip()],
        "final_class": str(payload.get("final_class") or "").strip(),
    }


def property_text(classification: Mapping[str, Any], fallback: str = "-") -> str:
    properties = classification.get("properties") or []
    values = [str(item).strip() for item in properties if str(item).strip()]
    return ", ".join(values) if values else fallback


def iter_prompt_lines(classification: Mapping[str, Any]) -> Iterable[str]:
    yield f"- Тип признаков: {str(classification.get('feature_type') or '').strip()}"
    yield f"- Структура: {str(classification.get('structure') or '').strip()}"
    yield f"- Свойства: {property_text(classification)}"

    final_class = str(classification.get("final_class") or "").strip()
    if final_class:
        yield f"- Финальная классификация: {final_class}"
