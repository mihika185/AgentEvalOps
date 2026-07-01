from datetime import datetime
from typing import Annotated, Any, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.models import (
    BenchmarkDataset,
    BenchmarkRun,
    BenchmarkRunItem,
    BenchmarkTestCase,
    Document,
    PipelineConfig,
    utc_now,
)
from backend.app.rag.answer_service import SimpleExtractiveAnswerGenerator
from backend.app.rag.workflow_service import RAGWorkflowError, run_rag_answer_workflow
from backend.app.evaluation.quality_gates import DEFAULT_QUALITY_GATE_PROFILE


router = APIRouter(
    prefix="/benchmarks",
    tags=["Benchmarks"]
)

ExpectedBehavior = Literal["answerable", "unanswerable"]

class BenchmarkDatasetCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    document_id: Optional[str] = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        return value.strip()

class BenchmarkDatasetUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    document_id: Optional[str] = None
    metadata_json: Optional[dict[str, Any]] = None

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if value is not None else value

class BenchmarkDatasetResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    document_id: Optional[str]
    metadata_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime

class BenchmarkTestCaseCreateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    document_id: Optional[str] = None
    expected_behavior: ExpectedBehavior
    expected_keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata_json: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        return value.strip()

    @field_validator("expected_keywords")
    @classmethod
    def clean_expected_keywords(cls, values: list[str]) -> list[str]:
        return [
            value.strip().lower()
            for value in values
            if value.strip()
        ]

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, values: list[str]) -> list[str]:
        return [
            value.strip().lower()
            for value in values
            if value.strip()
        ]

class BenchmarkTestCaseUpdateRequest(BaseModel):
    question: Optional[str] = Field(default=None, min_length=1)
    document_id: Optional[str] = None
    expected_behavior: Optional[ExpectedBehavior] = None
    expected_keywords: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    metadata_json: Optional[dict[str, Any]] = None

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if value is not None else value

    @field_validator("expected_keywords")
    @classmethod
    def clean_expected_keywords(
        cls,
        values: Optional[list[str]]
    ) -> Optional[list[str]]:
        if values is None:
            return None

        return [
            value.strip().lower()
            for value in values
            if value.strip()
        ]

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, values: Optional[list[str]]) -> Optional[list[str]]:
        if values is None:
            return None

        return [
            value.strip().lower()
            for value in values
            if value.strip()
        ]

class BenchmarkTestCaseResponse(BaseModel):
    id: str
    dataset_id: str
    question: str
    document_id: Optional[str]
    expected_behavior: str
    expected_keywords: list[str]
    tags: list[str]
    metadata_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime

class BenchmarkDatasetDetailResponse(BenchmarkDatasetResponse):
    test_cases: list[BenchmarkTestCaseResponse]

class BenchmarkRunItemResponse(BaseModel):
    id: str
    benchmark_run_id: str
    test_case_id: Optional[str]
    rag_run_id: Optional[str]
    question: str
    expected_behavior: str
    expected_keywords: list[str]
    actual_answer: Optional[str]
    passed: bool
    failure_reason: Optional[str]
    quality_gate_passed: bool
    response_blocked_by_quality_gate: bool
    metrics_json: dict[str, Any]
    source_chunks_json: list[dict[str, Any]]
    latency_ms: Optional[int]
    metadata_json: dict[str, Any]
    created_at: datetime

class BenchmarkRunResponse(BaseModel):
    id: str
    dataset_id: str
    status: str

    total_cases: int
    passed_cases: int
    failed_cases: int

    answerable_cases: int
    answerable_passed: int
    unanswerable_cases: int
    unanswerable_passed: int

    answerable_accuracy: Optional[float]
    unanswerable_accuracy: Optional[float]
    pass_rate: Optional[float]

    average_answer_support_score: Optional[float]
    average_query_answer_relevance_score: Optional[float]
    average_hallucination_risk: Optional[float]
    average_overall_quality_score: Optional[float]
    average_latency_ms: Optional[float]

    metadata_json: dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime]

class BenchmarkRunDetailResponse(BenchmarkRunResponse):
    items: list[BenchmarkRunItemResponse]

class BenchmarkComparisonRequest(BaseModel):
    pipeline_config_ids: list[str] = Field(..., min_length=1, max_length=5)

class PipelineBenchmarkResultResponse(BaseModel):
    pipeline_config_id: str
    pipeline_config_name: str
    benchmark_run: BenchmarkRunDetailResponse

class BenchmarkComparisonResponse(BaseModel):
    dataset_id: str
    total_pipeline_configs: int
    best_pipeline_config_id: Optional[str]
    best_pipeline_config_name: Optional[str]
    ranking_metric: str
    selection_reason: str
    results: list[PipelineBenchmarkResultResponse]

class BenchmarkFailureItemResponse(BaseModel):
    benchmark_item_id: str
    test_case_id: Optional[str]
    rag_run_id: Optional[str]
    question: str
    expected_behavior: str
    actual_answer: Optional[str]
    failure_reason: Optional[str]
    failure_category: str
    quality_gate_passed: bool
    response_blocked_by_quality_gate: bool
    metrics_json: dict[str, Any]
    pipeline_config_id: Optional[str]
    pipeline_config_name: Optional[str]

class BenchmarkFailureAnalysisResponse(BaseModel):
    benchmark_run_id: str
    dataset_id: str
    total_cases: int
    failed_cases: int
    failure_categories: dict[str, int]
    failed_items: list[BenchmarkFailureItemResponse]
    summary: str

def get_dataset_or_404(dataset_id: str, db: Session) -> BenchmarkDataset:
    dataset = db.get(BenchmarkDataset, dataset_id)

    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark dataset with id '{dataset_id}' was not found"
        )

    return dataset

def get_test_case_or_404(test_case_id: str, db: Session) -> BenchmarkTestCase:
    test_case = db.get(BenchmarkTestCase, test_case_id)

    if test_case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark test case with id '{test_case_id}' was not found"
        )

    return test_case

def ensure_document_exists(document_id: Optional[str], db: Session) -> None:
    if document_id is None:
        return

    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' was not found"
        )

def validate_test_case_payload(
    expected_behavior: str,
    expected_keywords: list[str],
    document_id: Optional[str]
) -> None:
    if document_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Test case needs a document_id either directly or from its dataset"
        )

    if expected_behavior == "answerable" and not expected_keywords:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Answerable test cases need at least one expected keyword"
        )

def answer_generator_from_pipeline_config(
    pipeline_config: PipelineConfig
):
    provider = pipeline_config.answer_generator_provider.strip().lower()

    if provider == "extractive":
        return SimpleExtractiveAnswerGenerator()

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            f"Unsupported answer_generator_provider "
            f"'{pipeline_config.answer_generator_provider}' for benchmark comparison"
        )
    )

def metadata_bool_value(
    metadata: Optional[dict[str, Any]],
    key: str,
    default: bool = False,
) -> bool:
    value = (metadata or {}).get(key, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return bool(value)

    if isinstance(value, str):
        cleaned_value = value.strip().lower()

        if cleaned_value in {"true", "1", "yes", "y"}:
            return True
        if cleaned_value in {"false", "0", "no", "n"}:
            return False

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"metadata_json.{key} must be a boolean value",
    )

def metadata_int_value(
    metadata: Optional[dict[str, Any]],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = (metadata or {}).get(key, default)

    try:
        parsed_value = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"metadata_json.{key} must be an integer value",
        ) from exc

    if parsed_value < minimum or parsed_value > maximum:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"metadata_json.{key} must be between "
                f"{minimum} and {maximum}"
            ),
        )

    return parsed_value

def execute_benchmark_run(
    db: Session,
    dataset: BenchmarkDataset,
    test_cases: list[BenchmarkTestCase],
    top_k: int,
    pipeline_config: Optional[PipelineConfig] = None
) -> tuple[BenchmarkRun, list[BenchmarkRunItem]]:
    resolved_quality_gate_profile = DEFAULT_QUALITY_GATE_PROFILE
    resolved_retrieval_provider = "dense"
    resolved_rerank = False
    resolved_candidate_multiplier = 3

    benchmark_metadata = {
        "dataset_name": dataset.name,
        "top_k": top_k,
        "retrieval_provider": resolved_retrieval_provider,
        "rerank": resolved_rerank,
        "candidate_multiplier": resolved_candidate_multiplier,
        "quality_gate_profile": resolved_quality_gate_profile,
        "runner_version": "benchmark-runner-v2"
    }

    answer_generator = None

    if pipeline_config is not None:
        resolved_quality_gate_profile = pipeline_config.quality_gate_profile
        resolved_retrieval_provider = pipeline_config.retrieval_provider
        resolved_rerank = metadata_bool_value(
            metadata=pipeline_config.metadata_json,
            key="rerank",
            default=False,
        )
        resolved_candidate_multiplier = metadata_int_value(
            metadata=pipeline_config.metadata_json,
            key="candidate_multiplier",
            default=3,
            minimum=1,
            maximum=10,
        )
        answer_generator = answer_generator_from_pipeline_config(pipeline_config)

        benchmark_metadata = {
            **benchmark_metadata,
            "pipeline_config_id": pipeline_config.id,
            "pipeline_config_name": pipeline_config.name,
            "retrieval_provider": pipeline_config.retrieval_provider,
            "answer_generator_provider": pipeline_config.answer_generator_provider,
            "answer_generator_model": pipeline_config.answer_generator_model,
            "embedding_provider": pipeline_config.embedding_provider,
            "embedding_model": pipeline_config.embedding_model,
            "quality_gate_profile": pipeline_config.quality_gate_profile,
            "rerank": resolved_rerank,
            "candidate_multiplier": resolved_candidate_multiplier,
            "reranker_settings_source": "pipeline_config.metadata_json",
        }

    benchmark_run = BenchmarkRun(
        dataset_id=dataset.id,
        status="running",
        total_cases=len(test_cases),
        metadata_json=benchmark_metadata
    )

    db.add(benchmark_run)
    db.commit()
    db.refresh(benchmark_run)

    run_items: list[BenchmarkRunItem] = []

    for test_case in test_cases:
        try:
            result = run_rag_answer_workflow(
                db=db,
                query=test_case.question,
                top_k=top_k,
                document_id=test_case.document_id or dataset.document_id,
                retrieval_provider=resolved_retrieval_provider,
                answer_generator=answer_generator,
                quality_gate_profile=resolved_quality_gate_profile,
                rerank=resolved_rerank,
                candidate_multiplier=resolved_candidate_multiplier,
            )

            metrics = to_metrics_dict(result)

            passed, failure_reason = judge_benchmark_case(
                expected_behavior=test_case.expected_behavior,
                expected_keywords=test_case.expected_keywords,
                answer=result.answer,
                quality_gate_passed=result.quality_gate_passed,
                response_blocked_by_quality_gate=result.response_blocked_by_quality_gate
            )

            run_item = BenchmarkRunItem(
                benchmark_run_id=benchmark_run.id,
                test_case_id=test_case.id,
                rag_run_id=result.run_id,
                question=test_case.question,
                expected_behavior=test_case.expected_behavior,
                expected_keywords=test_case.expected_keywords,
                actual_answer=result.answer,
                passed=passed,
                failure_reason=failure_reason,
                quality_gate_passed=result.quality_gate_passed,
                response_blocked_by_quality_gate=result.response_blocked_by_quality_gate,
                metrics_json=metrics,
                source_chunks_json=to_source_chunks_json(result),
                latency_ms=result.total_latency_ms,
                metadata_json={
                    "dataset_id": dataset.id,
                    "document_id": test_case.document_id or dataset.document_id,
                    "tags": test_case.tags,
                    "pipeline_config_id": pipeline_config.id if pipeline_config else None,
                    "pipeline_config_name": pipeline_config.name if pipeline_config else None,
                    "retrieval_provider": result.retrieval_provider,
                    "quality_gate_profile": result.quality_gate_profile,
                    "rerank": resolved_rerank,
                    "candidate_multiplier": resolved_candidate_multiplier,
                    "retrieved_chunk_count": result.retrieved_chunk_count,
                    "reranker_used": result.reranker_used,
                    "reranker_name": result.reranker_name,
                }
            )

        except RAGWorkflowError as exc:
            run_item = BenchmarkRunItem(
                benchmark_run_id=benchmark_run.id,
                test_case_id=test_case.id,
                rag_run_id=None,
                question=test_case.question,
                expected_behavior=test_case.expected_behavior,
                expected_keywords=test_case.expected_keywords,
                actual_answer=None,
                passed=False,
                failure_reason=str(exc),
                quality_gate_passed=False,
                response_blocked_by_quality_gate=False,
                metrics_json={},
                source_chunks_json=[],
                latency_ms=None,
                metadata_json={
                    "dataset_id": dataset.id,
                    "document_id": test_case.document_id or dataset.document_id,
                    "tags": test_case.tags,
                    "pipeline_config_id": pipeline_config.id if pipeline_config else None,
                    "pipeline_config_name": pipeline_config.name if pipeline_config else None,
                    "retrieval_provider": resolved_retrieval_provider,
                    "error_type": "RAGWorkflowError",
                    "quality_gate_profile": resolved_quality_gate_profile,
                    "rerank": resolved_rerank,
                    "candidate_multiplier": resolved_candidate_multiplier,
                    "reranker_used": False,
                    "reranker_name": None,
                }
            )

        db.add(run_item)
        run_items.append(run_item)

    db.commit()

    for item in run_items:
        db.refresh(item)

    finalize_benchmark_run(
        db=db,
        benchmark_run=benchmark_run,
        run_items=run_items
    )

    return benchmark_run, run_items

@router.post(
    "/datasets",
    response_model=BenchmarkDatasetResponse,
    status_code=status.HTTP_201_CREATED
)

def create_dataset(
    payload: BenchmarkDatasetCreateRequest,
    db: Annotated[Session, Depends(get_db)]
):
    ensure_document_exists(payload.document_id, db)

    dataset = BenchmarkDataset(
        name=payload.name,
        description=payload.description,
        document_id=payload.document_id,
        metadata_json=payload.metadata_json
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return to_dataset_response(dataset)

@router.get(
        "/datasets", 
        response_model=list[BenchmarkDatasetResponse]
)
def list_datasets(
    db: Annotated[Session, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20
):
    statement = (
        select(BenchmarkDataset)
        .order_by(BenchmarkDataset.created_at.desc())
        .offset(skip)
        .limit(limit)
    )

    datasets = db.execute(statement).scalars().all()

    return [
        to_dataset_response(dataset)
        for dataset in datasets
    ]

@router.get(
        "/datasets/{dataset_id}",
        response_model=BenchmarkDatasetDetailResponse
)
def get_dataset(
    dataset_id: str,
    db: Annotated[Session, Depends(get_db)]
):
    dataset = get_dataset_or_404(dataset_id, db)

    statement = (
        select(BenchmarkTestCase)
        .where(BenchmarkTestCase.dataset_id == dataset_id)
        .order_by(BenchmarkTestCase.created_at.asc())
    )

    test_cases = db.execute(statement).scalars().all()

    return BenchmarkDatasetDetailResponse(
        **to_dataset_response(dataset).model_dump(),
        test_cases=[
            to_test_case_response(test_case)
            for test_case in test_cases
        ]
    )

@router.patch(
        "/datasets/{dataset_id}",
        response_model=BenchmarkDatasetResponse
)
def update_dataset(
    dataset_id: str,
    payload: BenchmarkDatasetUpdateRequest,
    db: Annotated[Session, Depends(get_db)]
):
    dataset = get_dataset_or_404(dataset_id, db)

    update_data = payload.model_dump(exclude_unset=True)

    if "document_id" in update_data:
        ensure_document_exists(update_data["document_id"], db)

    for field, value in update_data.items():
        setattr(dataset, field, value)

    db.commit()
    db.refresh(dataset)

    return to_dataset_response(dataset)

@router.delete(
        "/datasets/{dataset_id}",
        status_code=status.HTTP_204_NO_CONTENT
)
def delete_dataset(
    dataset_id: str,
    db: Annotated[Session, Depends(get_db)]
):
    dataset = get_dataset_or_404(dataset_id, db)

    db.delete(dataset)
    db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.post(
    "/datasets/{dataset_id}/test-cases",
    response_model=BenchmarkTestCaseResponse,
    status_code=status.HTTP_201_CREATED
)
def create_test_case(
    dataset_id: str,
    payload: BenchmarkTestCaseCreateRequest,
    db: Annotated[Session, Depends(get_db)]
):
    dataset = get_dataset_or_404(dataset_id, db)

    resolved_document_id = payload.document_id or dataset.document_id

    ensure_document_exists(resolved_document_id, db)

    validate_test_case_payload(
        expected_behavior=payload.expected_behavior,
        expected_keywords=payload.expected_keywords,
        document_id=resolved_document_id
    )

    test_case = BenchmarkTestCase(
        dataset_id=dataset_id,
        question=payload.question,
        document_id=resolved_document_id,
        expected_behavior=payload.expected_behavior,
        expected_keywords=payload.expected_keywords,
        tags=payload.tags,
        metadata_json=payload.metadata_json
    )

    db.add(test_case)
    db.commit()
    db.refresh(test_case)

    return to_test_case_response(test_case)

@router.get(
    "/datasets/{dataset_id}/test-cases",
    response_model=list[BenchmarkTestCaseResponse]
)
def list_test_cases(
    dataset_id: str,
    db: Annotated[Session, Depends(get_db)]
):
    get_dataset_or_404(dataset_id, db)

    statement = (
        select(BenchmarkTestCase)
        .where(BenchmarkTestCase.dataset_id == dataset_id)
        .order_by(BenchmarkTestCase.created_at.asc())
    )

    test_cases = db.execute(statement).scalars().all()

    return [
        to_test_case_response(test_case)
        for test_case in test_cases
    ]

@router.patch(
        "/test-cases/{test_case_id}", 
        response_model=BenchmarkTestCaseResponse
)
def update_test_case(
    test_case_id: str,
    payload: BenchmarkTestCaseUpdateRequest,
    db: Annotated[Session, Depends(get_db)]
):
    test_case = get_test_case_or_404(test_case_id, db)

    update_data = payload.model_dump(exclude_unset=True)

    next_document_id = update_data.get("document_id", test_case.document_id)
    next_expected_behavior = update_data.get(
        "expected_behavior",
        test_case.expected_behavior
    )
    next_expected_keywords = update_data.get(
        "expected_keywords",
        test_case.expected_keywords
    )

    ensure_document_exists(next_document_id, db)

    validate_test_case_payload(
        expected_behavior=next_expected_behavior,
        expected_keywords=next_expected_keywords,
        document_id=next_document_id
    )

    for field, value in update_data.items():
        setattr(test_case, field, value)

    db.commit()
    db.refresh(test_case)

    return to_test_case_response(test_case)

@router.delete(
        "/test-cases/{test_case_id}",
        status_code=status.HTTP_204_NO_CONTENT
)
def delete_test_case(
    test_case_id: str,
    db: Annotated[Session, Depends(get_db)]
):
    test_case = get_test_case_or_404(test_case_id, db)

    db.delete(test_case)
    db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.post(
    "/datasets/{dataset_id}/runs",
    response_model=BenchmarkRunDetailResponse,
    status_code=status.HTTP_201_CREATED
)
def run_benchmark_dataset(
    dataset_id: str,
    db: Annotated[Session, Depends(get_db)],
    top_k: Annotated[int, Query(ge=1, le=20)] = 3
):
    dataset = get_dataset_or_404(dataset_id, db)

    test_cases = db.execute(
        select(BenchmarkTestCase)
        .where(BenchmarkTestCase.dataset_id == dataset_id)
        .order_by(BenchmarkTestCase.created_at.asc())
    ).scalars().all()

    if not test_cases:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Benchmark dataset has no test cases"
        )

    benchmark_run, run_items = execute_benchmark_run(
        db=db,
        dataset=dataset,
        test_cases=test_cases,
        top_k=top_k
    )

    return to_benchmark_run_detail_response(
        benchmark_run=benchmark_run,
        run_items=run_items
    )

@router.post(
    "/datasets/{dataset_id}/compare",
    response_model=BenchmarkComparisonResponse,
    status_code=status.HTTP_201_CREATED
)
def compare_pipeline_configs_on_dataset(
    dataset_id: str,
    payload: BenchmarkComparisonRequest,
    db: Annotated[Session, Depends(get_db)]
):
    dataset = get_dataset_or_404(dataset_id, db)

    test_cases = db.execute(
        select(BenchmarkTestCase)
        .where(BenchmarkTestCase.dataset_id == dataset_id)
        .order_by(BenchmarkTestCase.created_at.asc())
    ).scalars().all()

    if not test_cases:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Benchmark dataset has no test cases"
        )

    unique_config_ids = list(dict.fromkeys(payload.pipeline_config_ids))

    pipeline_configs: list[PipelineConfig] = []

    for pipeline_config_id in unique_config_ids:
        pipeline_config = db.get(PipelineConfig, pipeline_config_id)

        if pipeline_config is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline config with id '{pipeline_config_id}' was not found"
            )

        if not pipeline_config.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Pipeline config with id '{pipeline_config_id}' is not active"
            )

        pipeline_configs.append(pipeline_config)

    results: list[PipelineBenchmarkResultResponse] = []

    for pipeline_config in pipeline_configs:
        benchmark_run, run_items = execute_benchmark_run(
            db=db,
            dataset=dataset,
            test_cases=test_cases,
            top_k=pipeline_config.top_k,
            pipeline_config=pipeline_config
        )

        results.append(
            PipelineBenchmarkResultResponse(
                pipeline_config_id=pipeline_config.id,
                pipeline_config_name=pipeline_config.name,
                benchmark_run=to_benchmark_run_detail_response(
                    benchmark_run=benchmark_run,
                    run_items=run_items
                )
            )
        )

    best_result, selection_reason = pick_best_pipeline_result(results)

    return BenchmarkComparisonResponse(
        dataset_id=dataset_id,
        total_pipeline_configs=len(results),
        best_pipeline_config_id=best_result.pipeline_config_id if best_result else None,
        best_pipeline_config_name=best_result.pipeline_config_name if best_result else None,
        ranking_metric="pass_rate_then_overall_quality",
        selection_reason=selection_reason,
        results=results
    )

@router.get(
        "/runs",
        response_model=list[BenchmarkRunResponse]
)
def list_benchmark_runs(
    db: Annotated[Session, Depends(get_db)],
    dataset_id: Optional[str] = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20
):
    statement = select(BenchmarkRun)

    if dataset_id is not None:
        statement = statement.where(BenchmarkRun.dataset_id == dataset_id)

    statement = (
        statement
        .order_by(BenchmarkRun.started_at.desc())
        .offset(skip)
        .limit(limit)
    )

    runs = db.execute(statement).scalars().all()

    return [
        to_benchmark_run_response(run)
        for run in runs
    ]

@router.get(
    "/runs/{benchmark_run_id}", 
    response_model=BenchmarkRunDetailResponse
)

def get_benchmark_run(
    benchmark_run_id: str,
    db: Annotated[Session, Depends(get_db)]
):
    benchmark_run = db.get(BenchmarkRun, benchmark_run_id)

    if benchmark_run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark run with id '{benchmark_run_id}' was not found"
        )

    run_items = db.execute(
        select(BenchmarkRunItem)
        .where(BenchmarkRunItem.benchmark_run_id == benchmark_run_id)
        .order_by(BenchmarkRunItem.created_at.asc())
    ).scalars().all()

    return to_benchmark_run_detail_response(
        benchmark_run=benchmark_run,
        run_items=run_items
    )

@router.get(
    "/runs/{benchmark_run_id}/failure-analysis",
    response_model=BenchmarkFailureAnalysisResponse
)
def analyze_benchmark_run_failures(
    benchmark_run_id: str,
    db: Annotated[Session, Depends(get_db)]
):
    benchmark_run = db.get(BenchmarkRun, benchmark_run_id)

    if benchmark_run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark run with id '{benchmark_run_id}' was not found"
        )

    run_items = db.execute(
        select(BenchmarkRunItem)
        .where(BenchmarkRunItem.benchmark_run_id == benchmark_run_id)
        .order_by(BenchmarkRunItem.created_at.asc())
    ).scalars().all()

    failed_items = [
        item
        for item in run_items
        if not item.passed
    ]

    failure_responses = [
        to_failure_item_response(item)
        for item in failed_items
    ]

    failure_categories: dict[str, int] = {}

    for failure in failure_responses:
        failure_categories[failure.failure_category] = (
            failure_categories.get(failure.failure_category, 0) + 1
        )

    return BenchmarkFailureAnalysisResponse(
        benchmark_run_id=benchmark_run.id,
        dataset_id=benchmark_run.dataset_id,
        total_cases=len(run_items),
        failed_cases=len(failed_items),
        failure_categories=failure_categories,
        failed_items=failure_responses,
        summary=build_failure_summary(
            failed_cases=len(failed_items),
            failure_categories=failure_categories
        )
    )

def metric_value(metrics: dict[str, float], name: str) -> Optional[float]:
    value = metrics.get(name)

    if value is None:
        return None

    return float(value)

def average(values: list[float]) -> Optional[float]:
    if not values:
        return None

    return round(sum(values) / len(values), 4)

def keyword_missing_from_answer(
    answer: str,
    expected_keywords: list[str]
) -> list[str]:
    normalized_answer = answer.lower()

    return [
        keyword
        for keyword in expected_keywords
        if keyword.lower() not in normalized_answer
    ]

def judge_benchmark_case(
    expected_behavior: str,
    expected_keywords: list[str],
    answer: str,
    quality_gate_passed: bool,
    response_blocked_by_quality_gate: bool
) -> tuple[bool, Optional[str]]:
    if expected_behavior == "answerable":
        if response_blocked_by_quality_gate:
            return False, "Expected answerable response, but quality gate blocked the answer"

        if not quality_gate_passed:
            return False, "Expected answerable response, but quality gates did not pass"

        missing_keywords = keyword_missing_from_answer(
            answer=answer,
            expected_keywords=expected_keywords
        )

        if missing_keywords:
            return (
                False,
                f"Answer is missing expected keywords: {', '.join(missing_keywords)}"
            )

        return True, None

    if expected_behavior == "unanswerable":
        if response_blocked_by_quality_gate:
            return True, None

        return False, "Expected unanswerable query to be blocked, but answer was returned"

    return False, f"Unsupported expected behavior: {expected_behavior}"

def to_metrics_dict(result) -> dict[str, float]:
    return {
        metric.metric_name: metric.metric_value
        for metric in result.evaluation_metrics
    }

def pick_best_pipeline_result(
    results: list[PipelineBenchmarkResultResponse]
) -> tuple[Optional[PipelineBenchmarkResultResponse], str]:
    if not results:
        return None, "No pipeline results were available."

    def quality_key(result: PipelineBenchmarkResultResponse):
        run = result.benchmark_run

        return (
            run.pass_rate or 0.0,
            run.average_overall_quality_score or 0.0,
        )

    ranked_results = sorted(
        results,
        key=quality_key,
        reverse=True
    )

    best_result = ranked_results[0]
    best_key = quality_key(best_result)

    tied_results = [
        result
        for result in ranked_results
        if quality_key(result) == best_key
    ]

    if len(tied_results) > 1:
        tied_names = ", ".join(
            result.pipeline_config_name
            for result in tied_results
        )

        return (
            None,
            f"No clear winner. These configs tied on pass rate and quality: {tied_names}."
        )

    return (
        best_result,
        (
            "Selected because it had the highest pass rate, "
            "then highest average overall quality."
        )
    )

def categorize_benchmark_failure(item: BenchmarkRunItem) -> str:
    if item.passed:
        return "passed"

    failure_reason = (item.failure_reason or "").lower()

    if "missing expected keywords" in failure_reason:
        return "missing_expected_keywords"
    if "quality gate blocked" in failure_reason:
        return "answer_blocked_by_quality_gate"

    if "quality gates did not pass" in failure_reason:
        return "quality_gate_failed"

    if "unanswerable query to be blocked" in failure_reason:
        return "unanswerable_answer_returned"
    if item.metadata_json.get("error_type") == "RAGWorkflowError":
        return "rag_workflow_error"

    return "unknown_failure"

def build_failure_summary(
    failed_cases: int,
    failure_categories: dict[str, int]
) -> str:
    if failed_cases == 0:
        return "No benchmark failures were found for this run."

    sorted_categories = sorted(
        failure_categories.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_category, top_count = sorted_categories[0]

    if len(sorted_categories) == 1:
        return (
            f"{failed_cases} benchmark case(s) failed. "
            f"The only failure category was '{top_category}' "
            f"with {top_count} case(s)."
        )

    return (
        f"{failed_cases} benchmark case(s) failed. "
        f"The most common failure category was '{top_category}' "
        f"with {top_count} case(s)."
    )

def to_failure_item_response(
    item: BenchmarkRunItem
) -> BenchmarkFailureItemResponse:
    return BenchmarkFailureItemResponse(
        benchmark_item_id=item.id,
        test_case_id=item.test_case_id,
        rag_run_id=item.rag_run_id,
        question=item.question,
        expected_behavior=item.expected_behavior,
        actual_answer=item.actual_answer,
        failure_reason=item.failure_reason,
        failure_category=categorize_benchmark_failure(item),
        quality_gate_passed=item.quality_gate_passed,
        response_blocked_by_quality_gate=item.response_blocked_by_quality_gate,
        metrics_json=item.metrics_json,
        pipeline_config_id=item.metadata_json.get("pipeline_config_id"),
        pipeline_config_name=item.metadata_json.get("pipeline_config_name")
    )

def to_source_chunks_json(result) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "score": chunk.score,
            "text": chunk.text,
            "metadata": chunk.metadata
        }
        for chunk in result.source_chunks
    ]

def to_dataset_response(dataset: BenchmarkDataset) -> BenchmarkDatasetResponse:
    return BenchmarkDatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        document_id=dataset.document_id,
        metadata_json=dataset.metadata_json,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at
    )

def to_test_case_response(test_case: BenchmarkTestCase) -> BenchmarkTestCaseResponse:
    return BenchmarkTestCaseResponse(
        id=test_case.id,
        dataset_id=test_case.dataset_id,
        question=test_case.question,
        document_id=test_case.document_id,
        expected_behavior=test_case.expected_behavior,
        expected_keywords=test_case.expected_keywords,
        tags=test_case.tags,
        metadata_json=test_case.metadata_json,
        created_at=test_case.created_at,
        updated_at=test_case.updated_at
    )

def finalize_benchmark_run(
    db: Session,
    benchmark_run: BenchmarkRun,
    run_items: list[BenchmarkRunItem]
) -> None:
    passed_items = [
        item
        for item in run_items
        if item.passed
    ]

    answerable_items = [
        item
        for item in run_items
        if item.expected_behavior == "answerable"
    ]

    unanswerable_items = [
        item
        for item in run_items
        if item.expected_behavior == "unanswerable"
    ]

    answerable_passed = [
        item
        for item in answerable_items
        if item.passed
    ]

    unanswerable_passed = [
        item
        for item in unanswerable_items
        if item.passed
    ]

    support_scores = []
    relevance_scores = []
    hallucination_scores = []
    overall_scores = []
    latency_values = []

    for item in run_items:
        if item.latency_ms is not None:
            latency_values.append(float(item.latency_ms))

        if item.expected_behavior != "answerable":
            continue

        support_score = metric_value(item.metrics_json, "answer_support_score")
        relevance_score = metric_value(item.metrics_json, "query_answer_relevance_score")
        hallucination_risk = metric_value(item.metrics_json, "hallucination_risk")
        overall_score = metric_value(item.metrics_json, "overall_quality_score")

        if support_score is not None:
            support_scores.append(support_score)

        if relevance_score is not None:
            relevance_scores.append(relevance_score)

        if hallucination_risk is not None:
            hallucination_scores.append(hallucination_risk)

        if overall_score is not None:
            overall_scores.append(overall_score)

    benchmark_run.status = "completed"
    benchmark_run.total_cases = len(run_items)
    benchmark_run.passed_cases = len(passed_items)
    benchmark_run.failed_cases = len(run_items) - len(passed_items)

    benchmark_run.answerable_cases = len(answerable_items)
    benchmark_run.answerable_passed = len(answerable_passed)
    benchmark_run.unanswerable_cases = len(unanswerable_items)
    benchmark_run.unanswerable_passed = len(unanswerable_passed)

    benchmark_run.average_answer_support_score = average(support_scores)
    benchmark_run.average_query_answer_relevance_score = average(relevance_scores)
    benchmark_run.average_hallucination_risk = average(hallucination_scores)
    benchmark_run.average_overall_quality_score = average(overall_scores)
    benchmark_run.average_latency_ms = average(latency_values)

    benchmark_run.metadata_json = {
        **(benchmark_run.metadata_json or {}),
        "answer_quality_metrics_scope": "answerable_cases_only",
        "latency_metrics_scope": "all_cases",
        "unanswerable_success_metric": "response_blocked_by_quality_gate"
    }

    benchmark_run.completed_at = utc_now()

    db.commit()
    db.refresh(benchmark_run)

def to_benchmark_run_response(benchmark_run: BenchmarkRun) -> BenchmarkRunResponse:
    pass_rate = (
        benchmark_run.passed_cases / benchmark_run.total_cases
        if benchmark_run.total_cases
        else None
    )

    answerable_accuracy = (
        benchmark_run.answerable_passed / benchmark_run.answerable_cases
        if benchmark_run.answerable_cases
        else None
    )

    unanswerable_accuracy = (
        benchmark_run.unanswerable_passed / benchmark_run.unanswerable_cases
        if benchmark_run.unanswerable_cases
        else None
    )

    return BenchmarkRunResponse(
        id=benchmark_run.id,
        dataset_id=benchmark_run.dataset_id,
        status=benchmark_run.status,
        total_cases=benchmark_run.total_cases,
        passed_cases=benchmark_run.passed_cases,
        failed_cases=benchmark_run.failed_cases,
        answerable_cases=benchmark_run.answerable_cases,
        answerable_passed=benchmark_run.answerable_passed,
        unanswerable_cases=benchmark_run.unanswerable_cases,
        unanswerable_passed=benchmark_run.unanswerable_passed,
        answerable_accuracy=round(answerable_accuracy, 4)
        if answerable_accuracy is not None
        else None,
        unanswerable_accuracy=round(unanswerable_accuracy, 4)
        if unanswerable_accuracy is not None
        else None,
        pass_rate=round(pass_rate, 4)
        if pass_rate is not None
        else None,
        average_answer_support_score=benchmark_run.average_answer_support_score,
        average_query_answer_relevance_score=benchmark_run.average_query_answer_relevance_score,
        average_hallucination_risk=benchmark_run.average_hallucination_risk,
        average_overall_quality_score=benchmark_run.average_overall_quality_score,
        average_latency_ms=benchmark_run.average_latency_ms,
        metadata_json=benchmark_run.metadata_json,
        started_at=benchmark_run.started_at,
        completed_at=benchmark_run.completed_at
    )

def to_benchmark_run_item_response(
    item: BenchmarkRunItem
) -> BenchmarkRunItemResponse:
    return BenchmarkRunItemResponse(
        id=item.id,
        benchmark_run_id=item.benchmark_run_id,
        test_case_id=item.test_case_id,
        rag_run_id=item.rag_run_id,
        question=item.question,
        expected_behavior=item.expected_behavior,
        expected_keywords=item.expected_keywords,
        actual_answer=item.actual_answer,
        passed=item.passed,
        failure_reason=item.failure_reason,
        quality_gate_passed=item.quality_gate_passed,
        response_blocked_by_quality_gate=item.response_blocked_by_quality_gate,
        metrics_json=item.metrics_json,
        source_chunks_json=item.source_chunks_json,
        latency_ms=item.latency_ms,
        metadata_json=item.metadata_json,
        created_at=item.created_at
    )

def to_benchmark_run_detail_response(
    benchmark_run: BenchmarkRun,
    run_items: list[BenchmarkRunItem]
) -> BenchmarkRunDetailResponse:
    return BenchmarkRunDetailResponse(
        **to_benchmark_run_response(benchmark_run).model_dump(),
        items=[
            to_benchmark_run_item_response(item)
            for item in run_items
        ]
    )