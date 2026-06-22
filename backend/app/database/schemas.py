from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field

class APIBaseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

class DocumentCreate(APIBaseModel):
    filename: str
    file_type: str
    metadata_json: dict[str, Any] = Field(default_factory=dict)

class DocumentUpdate(APIBaseModel):
    filename: Optional[str] = None
    status: Optional[str] = None
    num_pages: Optional[int] = None
    num_chunks: Optional[int] = None
    metadata_json: Optional[dict[str, Any]] = None

class DocumentRead(APIBaseModel):
    id: str
    filename: str
    file_type: str
    status: str
    num_pages: Optional[int]
    num_chunks: int
    metadata_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime

class DocumentChunkCreate(APIBaseModel):
    document_id: str
    chunk_index: int
    chunk_text: str
    page_number: Optional[int] = None
    token_count: Optional[int] = None
    embedding_id: Optional[str] = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)

class DocumentChunkRead(APIBaseModel):
    id: str
    document_id: str
    chunk_index: int
    chunk_text: str
    page_number: Optional[int]
    token_count: Optional[int]
    embedding_id: Optional[str]
    metadata_json: dict[str, Any]
    created_at: datetime

class ExperimentCreate(APIBaseModel):
    name: str
    description: Optional[str] = None
    retriever_type: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    prompt_version: Optional[str] = None
    chunking_strategy: Optional[str] = None
    reranker_enabled: bool = False
    metadata_json: dict[str, Any] = Field(default_factory=dict)

class ExperimentRead(APIBaseModel):
    id: str
    name: str
    description: Optional[str]
    retriever_type: Optional[str]
    llm_provider: Optional[str]
    llm_model: Optional[str]
    prompt_version: Optional[str]
    chunking_strategy: Optional[str]
    reranker_enabled: bool
    metadata_json: dict[str, Any]
    created_at: datetime

class RunCreate(APIBaseModel):
    workflow_type: str
    input_query: str
    experiment_id: Optional[str] = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)

class RunUpdate(APIBaseModel):
    output_answer: Optional[str] = None
    status: Optional[str] = None
    latency_ms: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None
    metadata_json: Optional[dict[str, Any]] = None

class RunRead(APIBaseModel):
    id: str
    experiment_id: Optional[str]
    workflow_type: str
    input_query: str
    output_answer: Optional[str]
    status: str
    latency_ms: Optional[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    estimated_cost: Optional[float]
    error_message: Optional[str]
    metadata_json: dict[str, Any]
    created_at: datetime
    completed_at: Optional[datetime]

class TraceStepCreate(APIBaseModel):
    run_id: str
    step_index: int
    step_type: str
    name: str
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[int] = None
    status: str = "completed"
    error_message: Optional[str] = None

class TraceStepRead(APIBaseModel):
    id: str
    run_id: str
    step_index: int
    step_type: str
    name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    latency_ms: Optional[int]
    status: str
    error_message: Optional[str]
    created_at: datetime

class EvaluationResultCreate(APIBaseModel):
    run_id: str
    metric_name: str
    metric_value: float
    evaluator_type: str
    details_json: dict[str, Any] = Field(default_factory=dict)

class EvaluationResultRead(APIBaseModel):
    id: str
    run_id: str
    metric_name: str
    metric_value: float
    evaluator_type: str
    details_json: dict[str, Any]
    created_at: datetime

class QualityGateCreate(APIBaseModel):
    name: str
    metric_name: str
    operator: str
    threshold: float
    is_active: bool = True
    metadata_json: dict[str, Any] = Field(default_factory=dict)

class QualityGateUpdate(APIBaseModel):
    name: Optional[str] = None
    metric_name: Optional[str] = None
    operator: Optional[str] = None
    threshold: Optional[float] = None
    is_active: Optional[bool] = None
    metadata_json: Optional[dict[str, Any]] = None

class QualityGateRead(APIBaseModel):
    id: str
    name: str
    metric_name: str
    operator: str
    threshold: float
    is_active: bool
    metadata_json: dict[str, Any]
    created_at: datetime