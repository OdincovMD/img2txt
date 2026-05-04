"""OpenAI-compatible LLM client for clinical description generation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import httpx
from openai import OpenAI

from service.app.core.config import settings
from service.app.pipeline_steps.step4_generation.description_templates import format_classification

_client: OpenAI | None = None
_http_client: httpx.Client | None = None


def _build_http_client() -> httpx.Client:
    global _http_client
    if _http_client is not None:
        return _http_client

    client_kwargs: dict[str, Any] = {
        "timeout": settings.llm_timeout_seconds,
        "trust_env": True,
    }
    if settings.llm_proxy_url:
        client_kwargs["proxy"] = settings.llm_proxy_url

    _http_client = httpx.Client(**client_kwargs)
    return _http_client


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    api_key = settings.llm_api_key
    if not api_key:
        raise RuntimeError(
            "LLM API key is not set. Define LLM_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY."
        )

    _client = OpenAI(
        api_key=api_key,
        base_url=settings.llm_base_url,
        http_client=_build_http_client(),
    )
    return _client


def _build_single_prompt(features_text: str, classification: dict[str, Any] | str | None) -> str:
    prompt = "Составь клиническое дерматоскопическое описание по следующим данным:\n\n"
    prompt += features_text.strip()
    if classification is None:
        return prompt
    if isinstance(classification, dict):
        classification_text = format_classification(classification)
    else:
        classification_text = str(classification).strip()
    if classification_text:
        prompt += f"\n\n{classification_text}"
    return prompt


def _normalize_messages(
    features_text: str | Sequence[Mapping[str, str]],
    classification: dict[str, Any] | str | None,
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
    classification: dict[str, Any] | str | None = None,
    model: str | None = None,
    max_tokens: int = 2048,
) -> str:
    """Generate one Russian clinical description via an OpenAI-compatible API."""
    messages = _normalize_messages(features_text, classification)
    token_budget = max(1024, max_tokens)

    for attempt in range(2):
        response = _get_client().chat.completions.create(
            model=model or settings.description_model,
            messages=messages,
            temperature=0.3,
            top_p=0.85,
            max_completion_tokens=token_budget,
            extra_body={
                "reasoning_effort": "low",
                "include_reasoning": False,
            },
        )
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        finish_reason = getattr(choice, "finish_reason", None)

        if finish_reason != "length" and (not content or not content[-1].isalnum()):
            return content

        if attempt == 1:
            return content

        token_budget = max(token_budget * 2, 4096)

    return ""
