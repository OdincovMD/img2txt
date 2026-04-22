"""Groq-backed LLM client for clinical description generation."""

from __future__ import annotations

import os
from typing import Mapping, Sequence

from dotenv import load_dotenv
from openai import OpenAI

from service.app.schemas.classification import ClassificationResult
from service.app.pipeline_steps.step4_generation.description_templates import format_classification

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env or the environment.")

    _client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    return _client


def _build_single_prompt(features_text: str, classification: ClassificationResult | str | None) -> str:
    prompt = "Составь клиническое дерматоскопическое описание по следующим данным:\n\n"
    prompt += features_text.strip()
    if classification is None:
        return prompt
    if isinstance(classification, ClassificationResult):
        classification_text = format_classification(classification)
    else:
        classification_text = str(classification).strip()
    if classification_text:
        prompt += f"\n\n{classification_text}"
    return prompt


def _normalize_messages(
    features_text: str | Sequence[Mapping[str, str]],
    classification: ClassificationResult | str | None,
) -> list[dict[str, str]]:
    if isinstance(features_text, str):
        from service.app.pipeline_steps.step4_generation.description_templates import SYSTEM_PROMPT

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_single_prompt(features_text, classification)},
        ]

    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in features_text
    ]


def generate_description(
    features_text: str | Sequence[Mapping[str, str]],
    classification: ClassificationResult | str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 800,
) -> str:
    """Generate one Russian clinical description via Groq."""
    messages = _normalize_messages(features_text, classification)
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        top_p=0.85,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    return (content or "").strip()
