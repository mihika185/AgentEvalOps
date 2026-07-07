from typing import Annotated, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.reporting.aggregate_report_service import (
    AggregateReportError,
    build_benchmark_run_report,
    build_experiment_report,
)


router = APIRouter(
    prefix="/reports",
    tags=["Reports"],
)

class AggregateReportResponse(BaseModel):
    report_type: str
    scope_id: str
    generated_at: str
    service_version: str
    summary: dict[str, Any]
    quality_gate_result: dict[str, Any]
    readiness_decision: dict[str, Any]
    details: dict[str, Any]

@router.get(
    "/experiments/{experiment_id}",
    response_model=AggregateReportResponse,
)
def get_experiment_aggregate_report(
    experiment_id: str,
    db: Annotated[Session, Depends(get_db)],
    profile_name: str = Query(default="default-v1"),
    recent_run_limit: int = Query(default=10, ge=1, le=50),
):
    try:
        report = build_experiment_report(
            db=db,
            experiment_id=experiment_id,
            profile_name=profile_name,
            recent_run_limit=recent_run_limit,
        )
    except AggregateReportError as exc:
        http_status = (
            status.HTTP_404_NOT_FOUND
            if "was not found" in str(exc)
            else status.HTTP_400_BAD_REQUEST
        )

        raise HTTPException(
            status_code=http_status,
            detail=str(exc),
        ) from exc

    return AggregateReportResponse(**report)

@router.get(
    "/benchmark-runs/{benchmark_run_id}",
    response_model=AggregateReportResponse,
)
def get_benchmark_run_aggregate_report(
    benchmark_run_id: str,
    db: Annotated[Session, Depends(get_db)],
    profile_name: str = Query(default="default-v1"),
    failed_item_limit: int = Query(default=20, ge=1, le=100),
):
    try:
        report = build_benchmark_run_report(
            db=db,
            benchmark_run_id=benchmark_run_id,
            profile_name=profile_name,
            failed_item_limit=failed_item_limit,
        )
    except AggregateReportError as exc:
        http_status = (
            status.HTTP_404_NOT_FOUND
            if "was not found" in str(exc)
            else status.HTTP_400_BAD_REQUEST
        )

        raise HTTPException(
            status_code=http_status,
            detail=str(exc),
        ) from exc

    return AggregateReportResponse(**report)