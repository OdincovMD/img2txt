"""Clinical text generation step for one image."""

from __future__ import annotations

from typing import Any

from service.app.pipeline_steps.step4_generation.description_templates import (
    build_messages,
    parse_feature_entries,
)
from service.app.pipeline_steps.step4_generation.llm_client import generate_description


def generate_description_from_labels(
    important_labels: list[str],
    classification: dict[str, Any],
    model_name: str,
    max_tokens: int = 800,
) -> str:
    if not important_labels:
        return "Недостаточно данных для описания."
    grouped = parse_feature_entries(important_labels)
    messages = build_messages(grouped, classification)
    return generate_description(messages, model=model_name, max_tokens=max_tokens)
