from datetime import datetime
from typing import Annotated, Any, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database.connection import get_db
from backend.app.database.models import PipelineConfig
from backend.app.rag.llm_answer_generator import (
    LLMAnswerGenerationError,
    get_configured_answer_model as resolve_configured_answer_model,
    get_configured_answer_provider as resolve_configured_answer_provider,
)


router = APIRouter(
    prefix="/pipeline-configs",
    tags=["Pipeline Configs"]
)

AnswerGeneratorProvider = Literal["extractive", "groq"]
RetrievalProvider = Literal["dense", "bm25", "hybrid"]
QualityGateProfile = Literal["default-v1", "strict-v1", "lenient-v1"]

def get_configured_answer_provider() -> AnswerGeneratorProvider:
    try:
        return resolve_configured_answer_provider()
    except LLMAnswerGenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

def get_configured_answer_model() -> str:
    try:
        return resolve_configured_answer_model()
    except LLMAnswerGenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

def get_configured_embedding_provider() -> str:
    return settings.default_embedding_provider.strip()

def get_configured_embedding_model() -> str:
    return settings.default_embedding_model.strip()

class PipelineConfigCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None

    retrieval_provider: RetrievalProvider = "hybrid"
    top_k: int = Field(default=3, ge=1, le=20)

    answer_generator_provider: AnswerGeneratorProvider = Field(
        default_factory=get_configured_answer_provider
    )
    answer_generator_model: str = Field(
        default_factory=get_configured_answer_model
    )

    embedding_provider: str = Field(
        default_factory=get_configured_embedding_provider
    )
    embedding_model: str = Field(
        default_factory=get_configured_embedding_model
    )

    quality_gate_profile: QualityGateProfile = "default-v1"
    is_active: bool = True

    metadata_json: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        return value.strip()

    @field_validator(
        "retrieval_provider",
        "answer_generator_model",
        "embedding_provider",
        "embedding_model",
        "quality_gate_profile"
    )
    @classmethod
    def clean_string_field(cls, value: str) -> str:
        return value.strip()

class PipelineConfigUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None

    retrieval_provider: Optional[RetrievalProvider] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=20)

    answer_generator_provider: Optional[AnswerGeneratorProvider] = None
    answer_generator_model: Optional[str] = None

    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None

    quality_gate_profile: Optional[QualityGateProfile] = None
    is_active: Optional[bool] = None

    metadata_json: Optional[dict[str, Any]] = None

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if value is not None else value

    @field_validator(
        "retrieval_provider",
        "answer_generator_model",
        "embedding_provider",
        "embedding_model",
        "quality_gate_profile"
    )
    @classmethod
    def clean_string_field(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if value is not None else value

class PipelineConfigResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]

    retrieval_provider: str
    top_k: int

    answer_generator_provider: str
    answer_generator_model: str

    embedding_provider: str
    embedding_model: str

    quality_gate_profile: str
    is_active: bool

    metadata_json: dict[str, Any]

    created_at: datetime
    updated_at: datetime

class DefaultPipelineConfigsCreateResponse(BaseModel):
    created_count: int
    updated_count: int
    configs: list[PipelineConfigResponse]

def get_pipeline_config_or_404(
    pipeline_config_id: str,
    db: Session
) -> PipelineConfig:
    pipeline_config = db.get(PipelineConfig, pipeline_config_id)

    if pipeline_config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline config with id '{pipeline_config_id}' was not found"
        )

    return pipeline_config

@router.post(
    "/defaults",
    response_model=DefaultPipelineConfigsCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_default_pipeline_configs(
    db: Annotated[Session, Depends(get_db)],
):
    answer_provider = get_configured_answer_provider()
    answer_model = get_configured_answer_model()
    embedding_provider = get_configured_embedding_provider()
    embedding_model = get_configured_embedding_model()

    default_configs = [
        {
            "name": "BM25 Baseline",
            "description": "Keyword retrieval baseline without reranking.",
            "retrieval_provider": "bm25",
            "top_k": 3,
            "metadata_json": {
                "rerank": False,
                "candidate_multiplier": 3,
                "comparison_group": "default_retrieval_configs",
            },
        },
        {
            "name": "Dense Retrieval",
            "description": "Dense vector retrieval baseline without reranking.",
            "retrieval_provider": "dense",
            "top_k": 3,
            "metadata_json": {
                "rerank": False,
                "candidate_multiplier": 3,
                "comparison_group": "default_retrieval_configs",
            },
        },
        {
            "name": "Hybrid Retrieval",
            "description": "Hybrid BM25 plus dense retrieval without reranking.",
            "retrieval_provider": "hybrid",
            "top_k": 3,
            "metadata_json": {
                "rerank": False,
                "candidate_multiplier": 3,
                "comparison_group": "default_retrieval_configs",
            },
        },
        {
            "name": "Hybrid Retrieval + Rerank",
            "description": "Hybrid retrieval with cross-encoder reranking enabled.",
            "retrieval_provider": "hybrid",
            "top_k": 3,
            "metadata_json": {
                "rerank": True,
                "candidate_multiplier": 3,
                "comparison_group": "default_retrieval_configs",
            },
        },
    ]

    created_count = 0
    updated_count = 0

    for config_payload in default_configs:
        existing_config = db.execute(
            select(PipelineConfig)
            .where(PipelineConfig.name == config_payload["name"])
        ).scalar_one_or_none()

        if existing_config is not None:
            existing_config.description = config_payload["description"]
            existing_config.retrieval_provider = config_payload["retrieval_provider"]
            existing_config.top_k = config_payload["top_k"]
            existing_config.answer_generator_provider = answer_provider
            existing_config.answer_generator_model = answer_model
            existing_config.embedding_provider = embedding_provider
            existing_config.embedding_model = embedding_model
            existing_config.quality_gate_profile = "default-v1"
            existing_config.is_active = True
            existing_config.metadata_json = config_payload["metadata_json"]
            updated_count += 1
            continue

        pipeline_config = PipelineConfig(
            name=config_payload["name"],
            description=config_payload["description"],
            retrieval_provider=config_payload["retrieval_provider"],
            top_k=config_payload["top_k"],
            answer_generator_provider=answer_provider,
            answer_generator_model=answer_model,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            quality_gate_profile="default-v1",
            is_active=True,
            metadata_json=config_payload["metadata_json"],
        )

        db.add(pipeline_config)
        created_count += 1

    db.commit()

    configs = db.execute(
        select(PipelineConfig)
        .where(
            PipelineConfig.name.in_(
                [config_payload["name"] for config_payload in default_configs]
            )
        )
        .order_by(PipelineConfig.created_at.asc())
    ).scalars().all()

    return DefaultPipelineConfigsCreateResponse(
        created_count=created_count,
        updated_count=updated_count,
        configs=[
            to_pipeline_config_response(config)
            for config in configs
        ],
    )

@router.post(
    "",
    response_model=PipelineConfigResponse,
    status_code=status.HTTP_201_CREATED
)
def create_pipeline_config(
    payload: PipelineConfigCreateRequest,
    db: Annotated[Session, Depends(get_db)]
):
    pipeline_config = PipelineConfig(
        name=payload.name,
        description=payload.description,
        retrieval_provider=payload.retrieval_provider,
        top_k=payload.top_k,
        answer_generator_provider=payload.answer_generator_provider,
        answer_generator_model=payload.answer_generator_model,
        embedding_provider=payload.embedding_provider,
        embedding_model=payload.embedding_model,
        quality_gate_profile=payload.quality_gate_profile,
        is_active=payload.is_active,
        metadata_json=payload.metadata_json
    )

    db.add(pipeline_config)
    db.commit()
    db.refresh(pipeline_config)

    return to_pipeline_config_response(pipeline_config)

@router.get("", response_model=list[PipelineConfigResponse])
def list_pipeline_configs(
    db: Annotated[Session, Depends(get_db)],
    active_only: bool = True,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20
):
    statement = select(PipelineConfig)

    if active_only:
        statement = statement.where(PipelineConfig.is_active.is_(True))

    statement = (
        statement
        .order_by(PipelineConfig.created_at.desc())
        .offset(skip)
        .limit(limit)
    )

    pipeline_configs = db.execute(statement).scalars().all()

    return [
        to_pipeline_config_response(pipeline_config)
        for pipeline_config in pipeline_configs
    ]

@router.get("/{pipeline_config_id}", response_model=PipelineConfigResponse)
def get_pipeline_config(
    pipeline_config_id: str,
    db: Annotated[Session, Depends(get_db)]
):
    pipeline_config = get_pipeline_config_or_404(pipeline_config_id, db)

    return to_pipeline_config_response(pipeline_config)

@router.patch("/{pipeline_config_id}", response_model=PipelineConfigResponse)
def update_pipeline_config(
    pipeline_config_id: str,
    payload: PipelineConfigUpdateRequest,
    db: Annotated[Session, Depends(get_db)]
):
    pipeline_config = get_pipeline_config_or_404(pipeline_config_id, db)

    update_data = payload.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(pipeline_config, field, value)

    db.commit()
    db.refresh(pipeline_config)

    return to_pipeline_config_response(pipeline_config)

@router.delete("/{pipeline_config_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_pipeline_config(
    pipeline_config_id: str,
    db: Annotated[Session, Depends(get_db)]
):
    pipeline_config = get_pipeline_config_or_404(pipeline_config_id, db)

    db.delete(pipeline_config)
    db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)

def to_pipeline_config_response(
    pipeline_config: PipelineConfig
) -> PipelineConfigResponse:
    return PipelineConfigResponse(
        id=pipeline_config.id,
        name=pipeline_config.name,
        description=pipeline_config.description,
        retrieval_provider=pipeline_config.retrieval_provider,
        top_k=pipeline_config.top_k,
        answer_generator_provider=pipeline_config.answer_generator_provider,
        answer_generator_model=pipeline_config.answer_generator_model,
        embedding_provider=pipeline_config.embedding_provider,
        embedding_model=pipeline_config.embedding_model,
        quality_gate_profile=pipeline_config.quality_gate_profile,
        is_active=pipeline_config.is_active,
        metadata_json=pipeline_config.metadata_json,
        created_at=pipeline_config.created_at,
        updated_at=pipeline_config.updated_at
    )