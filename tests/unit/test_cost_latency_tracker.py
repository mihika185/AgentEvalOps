from types import SimpleNamespace

from backend.app.evaluation.cost_latency_tracker import (
    calculate_estimated_cost,
    estimate_token_count,
    track_answer_generation_usage,
)


class FakeGenerator:
    generator_name = "extractive:simple-extractive-v2"

class FakeProviderUsageGenerator:
    generator_name = "groq:llama-test"

    last_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "total_tokens": 125,
    }

def make_chunk(text: str):
    return SimpleNamespace(
        chunk_id="chunk_test",
        document_id="doc_test",
        text=text,
    )

def test_estimate_token_count_returns_zero_for_empty_text():
    assert estimate_token_count("") == 0

def test_estimate_token_count_returns_positive_count_for_text():
    assert estimate_token_count("Customers must report damaged products.") > 0

def test_track_answer_generation_usage_estimates_tokens_when_provider_usage_missing():
    usage = track_answer_generation_usage(
        generator=FakeGenerator(),
        generator_name=FakeGenerator.generator_name,
        query="What is the damaged product policy?",
        source_chunks=[
            make_chunk("Customers must report damaged products within 48 hours.")
        ],
        answer="Customers must report damaged products within 48 hours.",
        latency_ms=12,
    )

    assert usage.provider == "extractive"
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.estimated_cost == 0.0
    assert usage.token_usage_source == "estimated"
    assert usage.latency_ms == 12

def test_track_answer_generation_usage_uses_provider_usage_when_available():
    usage = track_answer_generation_usage(
        generator=FakeProviderUsageGenerator(),
        generator_name=FakeProviderUsageGenerator.generator_name,
        query="Question",
        source_chunks=[],
        answer="Answer",
        latency_ms=20,
    )

    assert usage.provider == "groq"
    assert usage.model == "llama-test"
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 25
    assert usage.total_tokens == 125
    assert usage.token_usage_source == "provider_usage"

def test_calculate_estimated_cost_uses_input_and_output_rates():
    cost = calculate_estimated_cost(
        prompt_tokens=1000,
        completion_tokens=500,
        input_cost_per_1k_tokens=0.20,
        output_cost_per_1k_tokens=0.60,
    )

    assert cost == 0.5