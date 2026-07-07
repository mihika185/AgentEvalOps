from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.reporting.dashboard_service import build_dashboard_summary


router = APIRouter(
    prefix="/dashboard",
    tags=["Dashboard"],
)

class DashboardSummaryResponse(BaseModel):
    service_version: str
    generated_at: str
    counts: dict[str, int]
    run_health: dict[str, Any]
    latency_cost: dict[str, Any]
    quality: dict[str, Any]
    recent_runs: list[dict[str, Any]]
    recent_experiments: list[dict[str, Any]]
    recent_benchmark_runs: list[dict[str, Any]]

@router.get(
    "/summary",
    response_model=DashboardSummaryResponse,
)
def get_dashboard_summary(
    db: Annotated[Session, Depends(get_db)],
    recent_limit: int = Query(default=8, ge=1, le=25),
):
    return DashboardSummaryResponse(
        **build_dashboard_summary(
            db=db,
            recent_limit=recent_limit,
        )
    )