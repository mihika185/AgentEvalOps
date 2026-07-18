from types import SimpleNamespace
import pytest

from backend.app.rag.answer_service import (
    SimpleExtractiveAnswerGenerator,
)
from backend.app.rag import (
    llm_answer_generator as generator_module,
)
from backend.app.rag.llm_answer_generator import (
    GroqGroundedAnswerGenerator,
    LLMAnswerGenerationError,
    create_answer_generator,
    get_configured_answer_model,
    get_configured_answer_provider,
    normalize_answer_provider,
    resolve_answer_model,
)
from backend.app.retrieval.retrieval_service import RetrievedChunk


@pytest.mark.parametrize("provider_name", ["mock", "simple", "extractive"])
def test_normalize_answer_provider_maps_local_aliases_to_extractive(provider_name):
    assert normalize_answer_provider(provider_name) == "extractive"

def test_normalize_answer_provider_accepts_groq_case_insensitively():
    assert normalize_answer_provider("  GrOq  ") == "groq"

def test_normalize_answer_provider_rejects_unsupported_provider():
    with pytest.raises(LLMAnswerGenerationError, match="Unsupported LLM provider"):
        normalize_answer_provider("unsupported")

def test_resolve_answer_model_uses_canonical_extractive_model():
    assert resolve_answer_model("mock", "ignored") == "simple-extractive-v2"

def test_resolve_answer_model_requires_hosted_model_name():
    with pytest.raises(
        LLMAnswerGenerationError,
        match="DEFAULT_LLM_MODEL must be set",
    ):
        resolve_answer_model("groq", "   ")

def test_create_answer_generator_returns_extractive_generator_for_local_provider():
    generator = create_answer_generator(
        provider_name="mock",
        model_name="mock-llm",
    )

    assert isinstance(generator, SimpleExtractiveAnswerGenerator)
    assert generator.generator_name == "simple-extractive-v2"

def test_create_answer_generator_passes_resolved_model_to_groq(monkeypatch):
    created_values = {}

    class FakeGroqGenerator:
        generator_name = "groq:test-model"

        def __init__(self, model_name, api_key=None):
            created_values["model_name"] = model_name
            created_values["api_key"] = api_key

    monkeypatch.setattr(
        generator_module,
        "GroqGroundedAnswerGenerator",
        FakeGroqGenerator,
    )

    generator = create_answer_generator(
        provider_name="groq",
        model_name=" test-model ",
        api_key="test-key",
    )

    assert isinstance(generator, FakeGroqGenerator)
    assert created_values == {
        "model_name": "test-model",
        "api_key": "test-key",
    }

def test_configured_answer_values_are_normalized(monkeypatch):
    monkeypatch.setattr(
        generator_module.settings,
        "default_llm_provider",
        "mock",
    )
    monkeypatch.setattr(
        generator_module.settings,
        "default_llm_model",
        "mock-llm",
    )

    assert get_configured_answer_provider() == "extractive"
    assert get_configured_answer_model() == "simple-extractive-v2"

def make_chunk(
    text: str = "The policy states the answer.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk_test",
        document_id="doc_test",
        score=1.0,
        text=text,
        metadata={},
    )

def make_completion(
    answer: str = "The policy states the answer.",
):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=6,
            total_tokens=26,
        ),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=answer)
            )
        ],
    )

def make_uninitialized_groq_generator(
    client,
    max_retries: int = 3,
):
    generator = GroqGroundedAnswerGenerator.__new__(
        GroqGroundedAnswerGenerator
    )

    generator.model_name = "test-model"
    generator.generator_name = "groq:test-model"
    generator.client = client
    generator.last_usage = None
    generator.last_retry_count = 0
    generator.last_retry_delays_seconds = []
    generator.last_retry_reasons = []
    generator.max_retries = max_retries
    generator.retry_base_seconds = 1.0
    generator.retry_max_seconds = 15.0
    generator.retry_jitter_seconds = 0.0

    return generator

def test_groq_generator_retries_429_using_retry_after(
    monkeypatch,
):
    class RetryResponse:
        headers = {"retry-after": "2.5"}

    class RetryableError(Exception):
        status_code = 429
        response = RetryResponse()

    class Completions:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            self.call_count += 1

            if self.call_count == 1:
                raise RetryableError("rate limited")

            return make_completion()

    completions = Completions()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    sleep_delays = []

    monkeypatch.setattr(
        generator_module.time,
        "sleep",
        sleep_delays.append,
    )

    generator = make_uninitialized_groq_generator(client)

    answer = generator.generate_answer(
        query="What does the policy state?",
        chunks=[make_chunk()],
    )

    assert answer == "The policy states the answer."
    assert completions.call_count == 2
    assert sleep_delays == [2.5]
    assert generator.last_retry_count == 1
    assert generator.last_retry_delays_seconds == [2.5]
    assert generator.last_retry_reasons == ["http_429"]

def test_groq_generator_does_not_retry_non_retryable_error(
    monkeypatch,
):
    class BadRequestError(Exception):
        status_code = 400
        response = None

    class Completions:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            self.call_count += 1
            raise BadRequestError("bad request")

    completions = Completions()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    monkeypatch.setattr(
        generator_module.time,
        "sleep",
        lambda delay: pytest.fail(
            "Non-retryable errors must not sleep"
        ),
    )

    generator = make_uninitialized_groq_generator(client)

    with pytest.raises(
        LLMAnswerGenerationError,
        match="after 1 attempt",
    ):
        generator.generate_answer(
            query="What does the policy state?",
            chunks=[make_chunk()],
        )

    assert completions.call_count == 1
    assert generator.last_retry_count == 0

def test_groq_generator_stops_after_retry_budget(
    monkeypatch,
):
    class RetryResponse:
        headers = {}

    class RetryableError(Exception):
        status_code = 503
        response = RetryResponse()

    class Completions:
        def __init__(self):
            self.call_count = 0

        def create(self, **kwargs):
            self.call_count += 1
            raise RetryableError("service unavailable")

    completions = Completions()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    sleep_delays = []

    monkeypatch.setattr(
        generator_module.time,
        "sleep",
        sleep_delays.append,
    )

    generator = make_uninitialized_groq_generator(
        client,
        max_retries=2,
    )

    with pytest.raises(
        LLMAnswerGenerationError,
        match="after 3 attempt",
    ):
        generator.generate_answer(
            query="What does the policy state?",
            chunks=[make_chunk()],
        )

    assert completions.call_count == 3
    assert sleep_delays == [1.0, 2.0]
    assert generator.last_retry_count == 2
    assert generator.last_retry_reasons == [
        "http_503",
        "http_503",
    ]