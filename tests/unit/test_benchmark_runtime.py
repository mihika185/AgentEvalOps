import pytest
from fastapi import HTTPException

from backend.app.api import benchmarks as benchmarks_module
from backend.app.api.benchmarks import (
    metadata_bool_value,
    metadata_int_value,
    resolve_benchmark_runtime,
)
from backend.app.config import settings
from backend.app.database.models import PipelineConfig
from backend.app.rag.answer_service import SimpleExtractiveAnswerGenerator
from backend.app.rag.llm_answer_generator import LLMAnswerGenerationError


class FakeAnswerGenerator:
    generator_name = "groq:llama-test"

    def generate_answer(self, query, chunks):
        return "answer"


def make_pipeline_config(**overrides):
    values = {
        "id": "pipe_test",
        "name": "Test Pipeline",
        "description": "Test configuration",
        "retrieval_provider": "hybrid",
        "top_k": 3,
        "answer_generator_provider": "groq",
        "answer_generator_model": "llama-test",
        "embedding_provider": "sentence-transformers",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "quality_gate_profile": "default-v1",
        "is_active": True,
        "metadata_json": {
            "rerank": True,
            "candidate_multiplier": 4,
        },
    }
    values.update(overrides)

    return PipelineConfig(**values)


def test_resolve_benchmark_runtime_uses_configured_defaults(monkeypatch):
    monkeypatch.setattr(settings, "default_llm_provider", "mock")
    monkeypatch.setattr(settings, "default_llm_model", "mock-llm")

    runtime = resolve_benchmark_runtime(None)

    assert runtime.retrieval_provider == "dense"
    assert runtime.answer_generator_provider == "extractive"
    assert runtime.answer_generator_model == "simple-extractive-v2"
    assert runtime.rerank is False
    assert runtime.candidate_multiplier == 3
    assert isinstance(runtime.answer_generator, SimpleExtractiveAnswerGenerator)


def test_resolve_benchmark_runtime_builds_pipeline_generator_once(monkeypatch):
    calls = []
    fake_generator = FakeAnswerGenerator()

    def fake_factory(pipeline_config):
        calls.append(pipeline_config.id)
        return fake_generator

    monkeypatch.setattr(
        benchmarks_module,
        "answer_generator_from_pipeline_config",
        fake_factory,
    )

    runtime = resolve_benchmark_runtime(make_pipeline_config())

    assert calls == ["pipe_test"]
    assert runtime.retrieval_provider == "hybrid"
    assert runtime.answer_generator_provider == "groq"
    assert runtime.answer_generator_model == "llama-test"
    assert runtime.rerank is True
    assert runtime.candidate_multiplier == 4
    assert runtime.answer_generator is fake_generator


def test_resolve_benchmark_runtime_rejects_unsupported_pipeline_provider():
    pipeline_config = make_pipeline_config(
        answer_generator_provider="unsupported",
    )

    with pytest.raises(HTTPException) as exc_info:
        resolve_benchmark_runtime(pipeline_config)

    assert exc_info.value.status_code == 400
    assert "Unsupported LLM provider" in exc_info.value.detail


def test_pipeline_generator_initialization_failure_returns_server_error(monkeypatch):
    def failing_factory(**kwargs):
        raise LLMAnswerGenerationError("GROQ_API_KEY is not set")

    monkeypatch.setattr(
        benchmarks_module,
        "create_answer_generator",
        failing_factory,
    )

    with pytest.raises(HTTPException) as exc_info:
        benchmarks_module.answer_generator_from_pipeline_config(
            make_pipeline_config()
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "GROQ_API_KEY is not set"


def test_metadata_bool_value_parses_supported_string_values():
    assert metadata_bool_value({"rerank": "yes"}, "rerank") is True
    assert metadata_bool_value({"rerank": "no"}, "rerank") is False


def test_metadata_bool_value_rejects_invalid_value():
    with pytest.raises(HTTPException) as exc_info:
        metadata_bool_value({"rerank": "sometimes"}, "rerank")

    assert exc_info.value.status_code == 400


def test_metadata_int_value_enforces_range():
    assert metadata_int_value(
        {"candidate_multiplier": "4"},
        "candidate_multiplier",
        default=3,
        minimum=1,
        maximum=10,
    ) == 4

    with pytest.raises(HTTPException) as exc_info:
        metadata_int_value(
            {"candidate_multiplier": 11},
            "candidate_multiplier",
            default=3,
            minimum=1,
            maximum=10,
        )

    assert exc_info.value.status_code == 400


def test_execute_benchmark_run_propagates_runtime_metadata(monkeypatch):
    from types import SimpleNamespace

    from backend.app.api.benchmarks import (
        BenchmarkRuntimeConfig,
        execute_benchmark_run,
    )

    fake_generator = FakeAnswerGenerator()
    runtime = BenchmarkRuntimeConfig(
        quality_gate_profile="strict-v1",
        retrieval_provider="hybrid",
        answer_generator_provider="groq",
        answer_generator_model="llama-test",
        embedding_provider="sentence-transformers",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rerank=True,
        candidate_multiplier=4,
        answer_generator=fake_generator,
    )
    workflow_calls = []

    class FakeDB:
        def __init__(self):
            self.added = []
            self.next_id = 1

        def add(self, value):
            if getattr(value, "id", None) is None:
                value.id = f"generated_{self.next_id}"
                self.next_id += 1
            self.added.append(value)

        def commit(self):
            pass

        def refresh(self, value):
            pass

    def fake_workflow(**kwargs):
        workflow_calls.append(kwargs)
        return SimpleNamespace(
            run_id="run_test",
            answer="The policy says 14 calendar days after delivery.",
            evaluation_metrics=[],
            source_chunks=[],
            quality_gate_passed=True,
            response_blocked_by_quality_gate=False,
            total_latency_ms=25,
            retrieval_provider="hybrid",
            answer_generator="groq:llama-test",
            quality_gate_profile="strict-v1",
            retrieved_chunk_count=0,
            reranker_used=True,
            reranker_name="cross-encoder-test",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            estimated_cost=0.0001,
        )

    monkeypatch.setattr(
        benchmarks_module,
        "resolve_benchmark_runtime",
        lambda pipeline_config: runtime,
    )
    monkeypatch.setattr(
        benchmarks_module,
        "run_rag_answer_workflow",
        fake_workflow,
    )
    monkeypatch.setattr(
        benchmarks_module,
        "finalize_benchmark_run",
        lambda **kwargs: None,
    )

    dataset = SimpleNamespace(
        id="dataset_test",
        name="Dataset",
        document_id="document_test",
    )
    test_case = SimpleNamespace(
        id="case_test",
        question="How long is the refund window?",
        document_id="document_test",
        expected_behavior="answerable",
        expected_keywords=["14 calendar days", "delivery"],
        tags=["refund"],
        metadata_json={},
    )
    pipeline_config = make_pipeline_config(
        quality_gate_profile="strict-v1",
    )

    benchmark_run, run_items = execute_benchmark_run(
        db=FakeDB(),
        dataset=dataset,
        test_cases=[test_case],
        top_k=3,
        pipeline_config=pipeline_config,
    )

    assert len(workflow_calls) == 1
    assert workflow_calls[0]["answer_generator"] is fake_generator
    assert workflow_calls[0]["retrieval_provider"] == "hybrid"
    assert workflow_calls[0]["rerank"] is True
    assert workflow_calls[0]["candidate_multiplier"] == 4
    assert benchmark_run.metadata_json["answer_generator_provider"] == "groq"
    assert benchmark_run.metadata_json["answer_generator_model"] == "llama-test"
    assert run_items[0].metadata_json["answer_generator_provider"] == "groq"
    assert run_items[0].metadata_json["answer_generator_model"] == "llama-test"
