from datetime import datetime
from typing import Annotated, Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.api.aggregate_quality_gate_models import (
    AggregateQualityGateSummaryResponse,
    to_aggregate_quality_gate_summary_response,
)
from backend.app.database.connection import get_db
from backend.app.database.models import EvaluationResult, Experiment, Run
from backend.app.evaluation.experiment_quality_gates import (
    AggregateQualityGateError,
    evaluate_experiment_quality_gates as run_experiment_quality_gates,
)


router = APIRouter(
    prefix="/experiments",
    tags=["Experiments"],
)

class ExperimentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    retriever_type: str = Field(default="hybrid", min_length=1, max_length=50)
    llm_provider: str = Field(default="mock", min_length=1, max_length=50)
    llm_model: str = Field(default="mock-llm", min_length=1, max_length=150)
    prompt_version: str = Field(default="v1", min_length=1, max_length=50)
    chunking_strategy: str = Field(default="default", min_length=1, max_length=100)
    reranker_enabled: bool = False
    metadata_json: dict[str, Any] = Field(default_factory=dict)

class ExperimentUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = None
    retriever_type: Optional[str] = Field(default=None, min_length=1, max_length=50)
    llm_provider: Optional[str] = Field(default=None, min_length=1, max_length=50)
    llm_model: Optional[str] = Field(default=None, min_length=1, max_length=150)
    prompt_version: Optional[str] = Field(default=None, min_length=1, max_length=50)
    chunking_strategy: Optional[str] = Field(default=None, min_length=1, max_length=100)
    reranker_enabled: Optional[bool] = None
    metadata_json: Optional[dict[str, Any]] = None

class ExperimentResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    retriever_type: str
    llm_provider: str
    llm_model: str
    prompt_version: str
    chunking_strategy: str
    reranker_enabled: bool
    metadata_json: dict[str, Any]
    run_count: int
    created_at: datetime

class ExperimentListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    experiments: list[ExperimentResponse]

class RunSummaryResponse(BaseModel):
    id: str
    experiment_id: Optional[str]
    workflow_type: str
    status: str
    input_query: Optional[str]
    output_answer: Optional[str]
    latency_ms: Optional[int]
    metadata_json: dict[str, Any]
    created_at: datetime

class ExperimentRunsResponse(BaseModel):
    experiment_id: str
    total: int
    limit: int
    offset: int
    runs: list[RunSummaryResponse]

class MetricSummaryResponse(BaseModel):
    evaluator_type: str
    metric_name: str
    count: int
    average_value: float
    min_value: float
    max_value: float

class ExperimentInspectionSummaryResponse(BaseModel):
    total_runs: int
    workflow_type_counts: dict[str, int]
    status_counts: dict[str, int]
    completed_runs: int
    failed_runs: int
    average_latency_ms: Optional[float]
    latest_run_created_at: Optional[datetime]
    total_evaluation_results: int
    metric_summaries: list[MetricSummaryResponse]

class ExperimentInspectionResponse(BaseModel):
    experiment: ExperimentResponse
    runs: list[RunSummaryResponse]
    summary: ExperimentInspectionSummaryResponse


@router.post(
    "",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_experiment(
    payload: ExperimentCreateRequest,
    db: Annotated[Session, Depends(get_db)],
):
    experiment = Experiment(
        id=create_experiment_id(),
        name=payload.name.strip(),
        description=clean_optional_string(payload.description),
        retriever_type=payload.retriever_type.strip(),
        llm_provider=payload.llm_provider.strip(),
        llm_model=payload.llm_model.strip(),
        prompt_version=payload.prompt_version.strip(),
        chunking_strategy=payload.chunking_strategy.strip(),
        reranker_enabled=payload.reranker_enabled,
        metadata_json=normalize_metadata_json(payload.metadata_json),
    )

    db.add(experiment)
    db.commit()
    db.refresh(experiment)

    return build_experiment_response(db, experiment)

@router.get("", response_model=ExperimentListResponse)
def list_experiments(
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    search: Optional[str] = Query(default=None),
):
    query = db.query(Experiment)

    cleaned_search = clean_optional_string(search)

    if cleaned_search:
        query = query.filter(Experiment.name.ilike(f"%{cleaned_search}%"))

    total = query.count()

    experiments = (
        query.order_by(Experiment.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return ExperimentListResponse(
        total=total,
        limit=limit,
        offset=offset,
        experiments=[
            build_experiment_response(db, experiment)
            for experiment in experiments
        ],
    )

@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(
    experiment_id: str,
    db: Annotated[Session, Depends(get_db)],
):
    experiment = get_experiment_or_404(db, experiment_id)

    return build_experiment_response(db, experiment)

@router.patch("/{experiment_id}", response_model=ExperimentResponse)
def update_experiment(
    experiment_id: str,
    payload: ExperimentUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
):
    experiment = get_experiment_or_404(db, experiment_id)

    update_data = payload.model_dump(exclude_unset=True)

    if "name" in update_data:
        experiment.name = update_data["name"].strip()

    if "description" in update_data:
        experiment.description = clean_optional_string(update_data["description"])

    if "retriever_type" in update_data:
        experiment.retriever_type = update_data["retriever_type"].strip()

    if "llm_provider" in update_data:
        experiment.llm_provider = update_data["llm_provider"].strip()

    if "llm_model" in update_data:
        experiment.llm_model = update_data["llm_model"].strip()

    if "prompt_version" in update_data:
        experiment.prompt_version = update_data["prompt_version"].strip()

    if "chunking_strategy" in update_data:
        experiment.chunking_strategy = update_data["chunking_strategy"].strip()

    if "reranker_enabled" in update_data:
        experiment.reranker_enabled = update_data["reranker_enabled"]

    if "metadata_json" in update_data:
        experiment.metadata_json = normalize_metadata_json(update_data["metadata_json"])

    db.commit()
    db.refresh(experiment)

    return build_experiment_response(db, experiment)

@router.post(
    "/{experiment_id}/runs/{run_id}",
    response_model=RunSummaryResponse,
)
def attach_run_to_experiment(
    experiment_id: str,
    run_id: str,
    db: Annotated[Session, Depends(get_db)],
):
    get_experiment_or_404(db, experiment_id)

    run = db.get(Run, run_id)

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run was not found: {run_id}",
        )

    run.experiment_id = experiment_id

    db.commit()
    db.refresh(run)

    return build_run_summary_response(run)

@router.get(
    "/{experiment_id}/runs",
    response_model=ExperimentRunsResponse,
)
def list_experiment_runs(
    experiment_id: str,
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    get_experiment_or_404(db, experiment_id)

    query = db.query(Run).filter(Run.experiment_id == experiment_id)

    total = query.count()

    runs = (
        query.order_by(Run.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return ExperimentRunsResponse(
        experiment_id=experiment_id,
        total=total,
        limit=limit,
        offset=offset,
        runs=[
            build_run_summary_response(run)
            for run in runs
        ],
    )

@router.get(
    "/{experiment_id}/inspection",
    response_model=ExperimentInspectionResponse,
)
def inspect_experiment(
    experiment_id: str,
    db: Annotated[Session, Depends(get_db)],
):
    experiment = get_experiment_or_404(db, experiment_id)

    runs = (
        db.query(Run)
        .filter(Run.experiment_id == experiment_id)
        .order_by(Run.created_at.desc())
        .all()
    )

    run_ids = [
        run.id
        for run in runs
    ]

    evaluation_results = []

    if run_ids:
        evaluation_results = (
            db.query(EvaluationResult)
            .filter(EvaluationResult.run_id.in_(run_ids))
            .all()
        )

    return ExperimentInspectionResponse(
        experiment=build_experiment_response(db, experiment),
        runs=[
            build_run_summary_response(run)
            for run in runs
        ],
        summary=build_experiment_inspection_summary(
            runs=runs,
            evaluation_results=evaluation_results,
        ),
    )

@router.post(
    "/{experiment_id}/quality-gates",
    response_model=AggregateQualityGateSummaryResponse,
)
def evaluate_experiment_aggregate_quality_gates(
    experiment_id: str,
    db: Annotated[Session, Depends(get_db)],
    profile_name: str = Query(default="default-v1"),
):
    try:
        summary = run_experiment_quality_gates(
            db=db,
            experiment_id=experiment_id,
            profile_name=profile_name,
            persist=True,
        )
    except AggregateQualityGateError as exc:
        http_status = (
            status.HTTP_404_NOT_FOUND
            if "was not found" in str(exc)
            else status.HTTP_400_BAD_REQUEST
        )

        raise HTTPException(
            status_code=http_status,
            detail=str(exc),
        ) from exc

    return to_aggregate_quality_gate_summary_response(summary)

def create_experiment_id() -> str:
    return f"exp_{uuid4().hex[:12]}"

def clean_optional_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    cleaned = value.strip()

    if not cleaned:
        return None

    return cleaned

def normalize_metadata_json(metadata_json: Optional[dict[str, Any]]) -> dict[str, Any]:
    if metadata_json is None:
        return {}

    return dict(metadata_json)

def get_experiment_or_404(db: Session, experiment_id: str) -> Experiment:
    experiment = db.get(Experiment, experiment_id)

    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment was not found: {experiment_id}",
        )

    return experiment

def build_experiment_response(
    db: Session,
    experiment: Experiment,
) -> ExperimentResponse:
    run_count = (
        db.query(Run)
        .filter(Run.experiment_id == experiment.id)
        .count()
    )

    return ExperimentResponse(
        id=experiment.id,
        name=experiment.name,
        description=experiment.description,
        retriever_type=experiment.retriever_type,
        llm_provider=experiment.llm_provider,
        llm_model=experiment.llm_model,
        prompt_version=experiment.prompt_version,
        chunking_strategy=experiment.chunking_strategy,
        reranker_enabled=experiment.reranker_enabled,
        metadata_json=experiment.metadata_json or {},
        run_count=run_count,
        created_at=experiment.created_at,
    )

def build_run_summary_response(run: Run) -> RunSummaryResponse:
    return RunSummaryResponse(
        id=run.id,
        experiment_id=run.experiment_id,
        workflow_type=run.workflow_type,
        status=run.status,
        input_query=getattr(run, "input_query", None),
        output_answer=getattr(run, "output_answer", None),
        latency_ms=getattr(run, "latency_ms", None),
        metadata_json=run.metadata_json or {},
        created_at=run.created_at,
    )

def count_by_field(items: list[Any], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}

    for item in items:
        raw_value = getattr(item, field_name, None)
        value = str(raw_value).strip() if raw_value is not None else "unknown"

        if not value:
            value = "unknown"

        counts[value] = counts.get(value, 0) + 1

    return dict(sorted(counts.items()))

def average_latency_ms(runs: list[Run]) -> Optional[float]:
    latency_values = [
        run.latency_ms
        for run in runs
        if run.latency_ms is not None
    ]

    if not latency_values:
        return None

    return round(sum(latency_values) / len(latency_values), 2)

def latest_run_created_at(runs: list[Run]) -> Optional[datetime]:
    created_at_values = [
        run.created_at
        for run in runs
        if run.created_at is not None
    ]

    if not created_at_values:
        return None

    return max(created_at_values)

def build_metric_summaries(
    evaluation_results: list[EvaluationResult],
) -> list[MetricSummaryResponse]:
    grouped_values: dict[tuple[str, str], list[float]] = {}

    for result in evaluation_results:
        key = (
            result.evaluator_type,
            result.metric_name,
        )

        grouped_values.setdefault(key, []).append(float(result.metric_value))

    summaries = []

    for key, values in grouped_values.items():
        evaluator_type, metric_name = key

        summaries.append(
            MetricSummaryResponse(
                evaluator_type=evaluator_type,
                metric_name=metric_name,
                count=len(values),
                average_value=round(sum(values) / len(values), 4),
                min_value=round(min(values), 4),
                max_value=round(max(values), 4),
            )
        )

    return sorted(
        summaries,
        key=lambda summary: (
            summary.evaluator_type,
            summary.metric_name,
        ),
    )

def build_experiment_inspection_summary(
    runs: list[Run],
    evaluation_results: list[EvaluationResult],
) -> ExperimentInspectionSummaryResponse:
    status_counts = count_by_field(runs, "status")

    return ExperimentInspectionSummaryResponse(
        total_runs=len(runs),
        workflow_type_counts=count_by_field(runs, "workflow_type"),
        status_counts=status_counts,
        completed_runs=status_counts.get("completed", 0),
        failed_runs=status_counts.get("failed", 0),
        average_latency_ms=average_latency_ms(runs),
        latest_run_created_at=latest_run_created_at(runs),
        total_evaluation_results=len(evaluation_results),
        metric_summaries=build_metric_summaries(evaluation_results),
    )