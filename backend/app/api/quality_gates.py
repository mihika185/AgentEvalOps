from datetime import datetime
from typing import Annotated, Any
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.models import QualityGate
from backend.app.evaluation.quality_gates import ensure_default_quality_gates


router = APIRouter(
    prefix="/quality-gates",
    tags=["Quality Gates"]
)

class QualityGateResponse(BaseModel):
    id: str
    name: str
    metric_name: str
    operator: str
    threshold: float
    is_active: bool
    metadata_json: dict[str, Any]
    created_at: datetime

class DefaultQualityGateCreateResponse(BaseModel):
    created_count: int

@router.post(
    "/defaults",
    response_model=DefaultQualityGateCreateResponse,
    status_code=status.HTTP_201_CREATED
)
def create_default_quality_gates(
    db: Annotated[Session, Depends(get_db)]
):
    created_count = ensure_default_quality_gates(db)

    return DefaultQualityGateCreateResponse(
        created_count=created_count
    )

@router.get("", response_model=list[QualityGateResponse])
def list_quality_gates(
    db: Annotated[Session, Depends(get_db)]
):
    statement = (
        select(QualityGate)
        .order_by(QualityGate.created_at.asc())
    )

    gates = db.execute(statement).scalars().all()

    return [
        QualityGateResponse(
            id=gate.id,
            name=gate.name,
            metric_name=gate.metric_name,
            operator=gate.operator,
            threshold=gate.threshold,
            is_active=gate.is_active,
            metadata_json=gate.metadata_json,
            created_at=gate.created_at
        )
        for gate in gates
    ]