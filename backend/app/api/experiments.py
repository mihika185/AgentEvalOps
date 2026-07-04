from datetime import datetime
from typing import Annotated, Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.models import Experiment, Run


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