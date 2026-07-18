from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.api.pipeline_configs import (
    PipelineConfigCreateRequest,
    create_default_pipeline_configs,
)
from backend.app.config import settings
from backend.app.database.connection import Base
from backend.app.database.models import PipelineConfig


EXPECTED_DEFAULT_PIPELINE_NAMES = {
    "BM25 Baseline",
    "Dense Retrieval",
    "Hybrid Retrieval",
    "Hybrid Retrieval + Rerank",
}


def test_pipeline_config_request_uses_configured_extractive_defaults(monkeypatch):
    monkeypatch.setattr(settings, "default_llm_provider", "mock")
    monkeypatch.setattr(settings, "default_llm_model", "mock-llm")

    payload = PipelineConfigCreateRequest(name="Test Config")

    assert payload.answer_generator_provider == "extractive"
    assert payload.answer_generator_model == "simple-extractive-v2"


def test_default_pipeline_configs_use_configured_provider_and_model(monkeypatch):
    monkeypatch.setattr(settings, "default_llm_provider", "groq")
    monkeypatch.setattr(settings, "default_llm_model", "llama-test")

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        response = create_default_pipeline_configs(db)
        configs = db.execute(select(PipelineConfig)).scalars().all()

    assert response.created_count == 4
    assert response.updated_count == 0
    assert {config.name for config in configs} == EXPECTED_DEFAULT_PIPELINE_NAMES
    assert all(config.answer_generator_provider == "groq" for config in configs)
    assert all(config.answer_generator_model == "llama-test" for config in configs)


def test_default_pipeline_configs_update_existing_records(monkeypatch):
    monkeypatch.setattr(settings, "default_llm_provider", "mock")
    monkeypatch.setattr(settings, "default_llm_model", "mock-llm")

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        create_default_pipeline_configs(db)

        monkeypatch.setattr(settings, "default_llm_provider", "groq")
        monkeypatch.setattr(settings, "default_llm_model", "llama-test")

        response = create_default_pipeline_configs(db)
        configs = db.execute(select(PipelineConfig)).scalars().all()

    assert response.created_count == 0
    assert response.updated_count == 4
    assert len(configs) == 4
    assert all(config.answer_generator_provider == "groq" for config in configs)
    assert all(config.answer_generator_model == "llama-test" for config in configs)
