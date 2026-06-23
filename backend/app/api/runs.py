from datetime import datetime
from typing import Annotated, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.models import EvaluationResult, Run, TraceStep
from backend.app.evaluation.answer_evaluator import EvaluationError, evaluate_rag_run


router = APIRouter(
    prefix="/runs",
    tags=["Runs"]
)

class RunSummaryResponse(BaseModel):
    id: str
    workflow_type: str
    status: str
    input_query: str
    output_answer: Optional[str]
    latency_ms: Optional[int]
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    metadata_json: dict[str, Any]

class RunDetailResponse(BaseModel):
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

class TraceStepResponse(BaseModel):
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

class MetricResponse(BaseModel):
    metric_name: str
    metric_value: float
    details: dict[str, Any]

class RunEvaluationResponse(BaseModel):
    run_id: str
    evaluator_type: str
    metrics: list[MetricResponse]

class EvaluationResultResponse(BaseModel):
    id: str
    run_id: str
    metric_name: str
    metric_value: float
    evaluator_type: str
    details_json: dict[str, Any]
    created_at: datetime

def get_run_or_404(run_id: str, db: Session) -> Run:
    run = db.get(Run, run_id)

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run with id '{run_id}' was not found"
        )

    return run

@router.get("", response_model=list[RunSummaryResponse])
def list_runs(
    db: Annotated[Session, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    workflow_type: Optional[str] = None,
    status_filter: Optional[str] = Query(default=None, alias="status")
):
    statement = select(Run)

    if workflow_type is not None:
        statement = statement.where(Run.workflow_type == workflow_type)

    if status_filter is not None:
        statement = statement.where(Run.status == status_filter)

    statement = (
        statement
        .order_by(Run.created_at.desc())
        .offset(skip)
        .limit(limit)
    )

    runs = db.execute(statement).scalars().all()

    return [
        RunSummaryResponse(
            id=run.id,
            workflow_type=run.workflow_type,
            status=run.status,
            input_query=run.input_query,
            output_answer=run.output_answer,
            latency_ms=run.latency_ms,
            error_message=run.error_message,
            created_at=run.created_at,
            completed_at=run.completed_at,
            metadata_json=run.metadata_json
        )
        for run in runs
    ]

@router.get("/{run_id}", response_model=RunDetailResponse)
def get_run(
    run_id: str,
    db: Annotated[Session, Depends(get_db)]
):
    run = get_run_or_404(run_id, db)

    return RunDetailResponse(
        id=run.id,
        experiment_id=run.experiment_id,
        workflow_type=run.workflow_type,
        input_query=run.input_query,
        output_answer=run.output_answer,
        status=run.status,
        latency_ms=run.latency_ms,
        prompt_tokens=run.prompt_tokens,
        completion_tokens=run.completion_tokens,
        estimated_cost=run.estimated_cost,
        error_message=run.error_message,
        metadata_json=run.metadata_json,
        created_at=run.created_at,
        completed_at=run.completed_at
    )

@router.get("/{run_id}/trace", response_model=list[TraceStepResponse])
def get_run_trace(
    run_id: str,
    db: Annotated[Session, Depends(get_db)]
):
    get_run_or_404(run_id, db)

    statement = (
        select(TraceStep)
        .where(TraceStep.run_id == run_id)
        .order_by(TraceStep.step_index.asc())
    )

    trace_steps = db.execute(statement).scalars().all()

    return [
        TraceStepResponse(
            id=step.id,
            run_id=step.run_id,
            step_index=step.step_index,
            step_type=step.step_type,
            name=step.name,
            input_data=step.input_data,
            output_data=step.output_data,
            latency_ms=step.latency_ms,
            status=step.status,
            error_message=step.error_message,
            created_at=step.created_at
        )
        for step in trace_steps
    ]

@router.post("/{run_id}/evaluate", response_model=RunEvaluationResponse)
def evaluate_run(
    run_id: str,
    db: Annotated[Session, Depends(get_db)]
):
    get_run_or_404(run_id, db)

    try:
        summary = evaluate_rag_run(db=db, run_id=run_id, persist=True)

        return RunEvaluationResponse(
            run_id=summary.run_id,
            evaluator_type=summary.evaluator_type,
            metrics=[
                MetricResponse(
                    metric_name=metric.metric_name,
                    metric_value=metric.metric_value,
                    details=metric.details
                )
                for metric in summary.metrics
            ]
        )

    except EvaluationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc

@router.get("/{run_id}/evaluations", response_model=list[EvaluationResultResponse])
def get_run_evaluations(
    run_id: str,
    db: Annotated[Session, Depends(get_db)]
):
    get_run_or_404(run_id, db)

    statement = (
        select(EvaluationResult)
        .where(EvaluationResult.run_id == run_id)
        .order_by(EvaluationResult.metric_name.asc())
    )

    evaluation_results = db.execute(statement).scalars().all()

    return [
        EvaluationResultResponse(
            id=result.id,
            run_id=result.run_id,
            metric_name=result.metric_name,
            metric_value=result.metric_value,
            evaluator_type=result.evaluator_type,
            details_json=result.details_json,
            created_at=result.created_at
        )
        for result in evaluation_results
    ]