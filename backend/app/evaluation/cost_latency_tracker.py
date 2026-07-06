import math
from dataclasses import dataclass
from typing import Any, Optional

from backend.app.config import settings


@dataclass(frozen=True)
class GenerationUsage:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float
    token_usage_source: str
    latency_ms: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
            "input_cost_per_1k_tokens": self.input_cost_per_1k_tokens,
            "output_cost_per_1k_tokens": self.output_cost_per_1k_tokens,
            "token_usage_source": self.token_usage_source,
            "latency_ms": self.latency_ms,
        }


def track_answer_generation_usage(
    generator: Any,
    generator_name: str,
    query: str,
    source_chunks: list[Any],
    answer: str,
    latency_ms: int,
) -> GenerationUsage:
    provider, model = parse_generator_name(generator_name)
    provider_usage = get_provider_usage(generator)

    if provider_usage is not None:
        prompt_tokens = provider_usage["prompt_tokens"]
        completion_tokens = provider_usage["completion_tokens"]
        token_usage_source = "provider_usage"
    else:
        prompt_tokens = estimate_prompt_tokens(
            query=query,
            source_chunks=source_chunks,
        )
        completion_tokens = estimate_token_count(answer)
        token_usage_source = "estimated"

    total_tokens = prompt_tokens + completion_tokens

    input_cost_per_1k_tokens, output_cost_per_1k_tokens = get_provider_cost_rates(
        provider
    )

    estimated_cost = calculate_estimated_cost(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        input_cost_per_1k_tokens=input_cost_per_1k_tokens,
        output_cost_per_1k_tokens=output_cost_per_1k_tokens,
    )

    return GenerationUsage(
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost=estimated_cost,
        input_cost_per_1k_tokens=input_cost_per_1k_tokens,
        output_cost_per_1k_tokens=output_cost_per_1k_tokens,
        token_usage_source=token_usage_source,
        latency_ms=max(int(latency_ms), 0),
    )


def get_provider_usage(generator: Any) -> Optional[dict[str, int]]:
    raw_usage = getattr(generator, "last_usage", None)

    if not raw_usage:
        return None

    prompt_tokens = get_int_value(raw_usage, "prompt_tokens")
    completion_tokens = get_int_value(raw_usage, "completion_tokens")

    if prompt_tokens is None or completion_tokens is None:
        return None

    return {
        "prompt_tokens": max(prompt_tokens, 0),
        "completion_tokens": max(completion_tokens, 0),
    }


def get_int_value(data: dict[str, Any], key: str) -> Optional[int]:
    value = data.get(key)

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_generator_name(generator_name: str) -> tuple[str, str]:
    cleaned_name = str(generator_name or "").strip()

    if not cleaned_name:
        return "unknown", "unknown"

    if ":" in cleaned_name:
        provider, model = cleaned_name.split(":", 1)
        return provider.strip().lower(), model.strip()

    if cleaned_name.startswith("simple-extractive"):
        return "extractive", cleaned_name

    return "unknown", cleaned_name


def estimate_prompt_tokens(query: str, source_chunks: list[Any]) -> int:
    prompt_parts = [
        "Question:",
        query,
        "Retrieved context:",
    ]

    for index, chunk in enumerate(source_chunks, start=1):
        chunk_id = str(getattr(chunk, "chunk_id", getattr(chunk, "id", "unknown")))
        document_id = str(getattr(chunk, "document_id", "unknown"))
        text = str(getattr(chunk, "text", getattr(chunk, "chunk_text", "")))

        prompt_parts.extend(
            [
                f"[Source {index}]",
                f"chunk_id: {chunk_id}",
                f"document_id: {document_id}",
                text,
            ]
        )

    prompt_parts.extend(
        [
            "Instructions:",
            "Answer using only the retrieved context.",
            "Do not invent facts.",
            "If the answer is not supported, use the fallback sentence.",
        ]
    )

    return estimate_token_count("\n".join(prompt_parts))


def estimate_token_count(text: str) -> int:
    cleaned_text = str(text or "").strip()

    if not cleaned_text:
        return 0

    chars_per_token = max(settings.token_estimation_chars_per_token, 1)

    return max(1, math.ceil(len(cleaned_text) / chars_per_token))


def get_provider_cost_rates(provider: str) -> tuple[float, float]:
    normalized_provider = provider.strip().lower()

    if normalized_provider == "groq":
        return (
            settings.groq_input_cost_per_1k_tokens,
            settings.groq_output_cost_per_1k_tokens,
        )

    if normalized_provider == "openai":
        return (
            settings.openai_input_cost_per_1k_tokens,
            settings.openai_output_cost_per_1k_tokens,
        )

    return (
        settings.default_input_cost_per_1k_tokens,
        settings.default_output_cost_per_1k_tokens,
    )


def calculate_estimated_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_cost_per_1k_tokens: float,
    output_cost_per_1k_tokens: float,
) -> float:
    input_cost = (prompt_tokens / 1000.0) * input_cost_per_1k_tokens
    output_cost = (completion_tokens / 1000.0) * output_cost_per_1k_tokens

    return round(input_cost + output_cost, 8)