from functools import lru_cache
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_name: str = "AgentEvalOps API"
    app_description: str = (
        "AI reliability platform for evaluating RAG pipelines "
        "and tool-calling agents."
    )
    app_version: str = "0.1.0"

    environment: Literal["development", "testing", "production"] = "development"
    debug: bool = True
    api_prefix: str = "/api/v1"

    database_url: str = (
        "postgresql://agentevalops:agentevalops@localhost:5432/agentevalops"
    )
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"

    default_llm_provider: str = "mock"
    default_llm_model: str = "mock-llm"
    default_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

    max_upload_mb: int = 25
    default_retrieval_top_k: int = 5
    max_retrieval_top_k: int = 20

    enable_trace_logging: bool = True

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_testing(self) -> bool:
        return self.environment == "testing"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()