from datetime import datetime
from typing import Annotated, Any, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.models import QualityGate
from backend.app.evaluation.quality_gates import ensure_default_quality_gates


router = APIRouter(
    prefix="/quality-gates",
    tags=["Quality Gates"],
)

GateOperator = Literal[">=", "<=", ">", "<", "=="]


class QualityGateCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    metric_name: str = Field(..., min_length=1, max_length=120)
    operator: GateOperator
    threshold: float
    is_active: bool = True
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class QualityGateUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    metric_name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    operator: Optional[GateOperator] = None
    threshold: Optional[float] = None
    is_active: Optional[bool] = None
    metadata_json: Optional[dict[str, Any]] = None


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


class QualityGateMetricsCheckRequest(BaseModel):
    metrics: dict[str, float]
    active_only: bool = True
    fail_on_missing_metrics: bool = True


class QualityGateCheckItemResponse(BaseModel):
    gate_id: str
    name: str
    metric_name: str
    operator: str
    threshold: float
    actual_value: Optional[float]
    passed: bool
    reason: str


class QualityGateMetricsCheckResponse(BaseModel):
    gate_status: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    checks: list[QualityGateCheckItemResponse]


@router.post(
    "/defaults",
    response_model=DefaultQualityGateCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_default_quality_gates(
    db: Annotated[Session, Depends(get_db)],
):
    created_count = ensure_default_quality_gates(db)

    return DefaultQualityGateCreateResponse(
        created_count=created_count,
    )


@router.post(
    "",
    response_model=QualityGateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_quality_gate(
    payload: QualityGateCreateRequest,
    db: Annotated[Session, Depends(get_db)],
):
    gate = QualityGate(
        id=create_quality_gate_id(),
        name=payload.name.strip(),
        metric_name=payload.metric_name.strip(),
        operator=payload.operator,
        threshold=payload.threshold,
        is_active=payload.is_active,
        metadata_json=payload.metadata_json,
    )

    db.add(gate)
    db.commit()
    db.refresh(gate)

    return to_quality_gate_response(gate)


@router.get("", response_model=list[QualityGateResponse])
def list_quality_gates(
    db: Annotated[Session, Depends(get_db)],
):
    statement = (
        select(QualityGate)
        .order_by(QualityGate.created_at.asc())
    )

    gates = db.execute(statement).scalars().all()

    return [
        to_quality_gate_response(gate)
        for gate in gates
    ]


@router.get("/{gate_id}", response_model=QualityGateResponse)
def get_quality_gate(
    gate_id: str,
    db: Annotated[Session, Depends(get_db)],
):
    gate = get_quality_gate_or_404(gate_id=gate_id, db=db)

    return to_quality_gate_response(gate)


@router.patch("/{gate_id}", response_model=QualityGateResponse)
def update_quality_gate(
    gate_id: str,
    payload: QualityGateUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
):
    gate = get_quality_gate_or_404(gate_id=gate_id, db=db)

    update_data = payload.model_dump(exclude_unset=True)

    if "name" in update_data:
        gate.name = update_data["name"].strip()

    if "metric_name" in update_data:
        gate.metric_name = update_data["metric_name"].strip()

    if "operator" in update_data:
        gate.operator = update_data["operator"]

    if "threshold" in update_data:
        gate.threshold = update_data["threshold"]

    if "is_active" in update_data:
        gate.is_active = update_data["is_active"]

    if "metadata_json" in update_data:
        gate.metadata_json = update_data["metadata_json"] or {}

    db.commit()
    db.refresh(gate)

    return to_quality_gate_response(gate)


@router.delete("/{gate_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_quality_gate(
    gate_id: str,
    db: Annotated[Session, Depends(get_db)],
):
    gate = get_quality_gate_or_404(gate_id=gate_id, db=db)

    db.delete(gate)
    db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/check", response_model=QualityGateMetricsCheckResponse)
def check_metrics_against_quality_gates(
    payload: QualityGateMetricsCheckRequest,
    db: Annotated[Session, Depends(get_db)],
):
    statement = select(QualityGate).order_by(QualityGate.created_at.asc())

    if payload.active_only:
        statement = statement.where(QualityGate.is_active.is_(True))

    gates = db.execute(statement).scalars().all()

    checks = []

    for gate in gates:
        if gate.metric_name not in payload.metrics and not payload.fail_on_missing_metrics:
            continue

        checks.append(
            check_quality_gate(
                gate=gate,
                metrics=payload.metrics,
            )
        )

    passed_checks = sum(1 for check in checks if check.passed)
    failed_checks = len(checks) - passed_checks

    return QualityGateMetricsCheckResponse(
        gate_status="passed" if failed_checks == 0 else "failed",
        total_checks=len(checks),
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        checks=checks,
    )


def create_quality_gate_id() -> str:
    return f"gate_{uuid4().hex[:12]}"


def get_quality_gate_or_404(gate_id: str, db: Session) -> QualityGate:
    gate = db.get(QualityGate, gate_id)

    if gate is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quality gate was not found: {gate_id}",
        )

    return gate


def to_quality_gate_response(gate: QualityGate) -> QualityGateResponse:
    return QualityGateResponse(
        id=gate.id,
        name=gate.name,
        metric_name=gate.metric_name,
        operator=gate.operator,
        threshold=gate.threshold,
        is_active=gate.is_active,
        metadata_json=gate.metadata_json or {},
        created_at=gate.created_at,
    )


def check_quality_gate(
    gate: QualityGate,
    metrics: dict[str, float],
) -> QualityGateCheckItemResponse:
    actual_value = metrics.get(gate.metric_name)

    if actual_value is None:
        return QualityGateCheckItemResponse(
            gate_id=gate.id,
            name=gate.name,
            metric_name=gate.metric_name,
            operator=gate.operator,
            threshold=gate.threshold,
            actual_value=None,
            passed=False,
            reason=f"Metric '{gate.metric_name}' was missing from payload.",
        )

    passed = compare_metric(
        actual_value=float(actual_value),
        operator=gate.operator,
        threshold=float(gate.threshold),
    )

    reason = (
        f"{gate.metric_name} {gate.operator} {gate.threshold} passed "
        f"with actual value {actual_value}."
        if passed
        else f"{gate.metric_name} {gate.operator} {gate.threshold} failed "
        f"with actual value {actual_value}."
    )

    return QualityGateCheckItemResponse(
        gate_id=gate.id,
        name=gate.name,
        metric_name=gate.metric_name,
        operator=gate.operator,
        threshold=gate.threshold,
        actual_value=float(actual_value),
        passed=passed,
        reason=reason,
    )


def compare_metric(
    actual_value: float,
    operator: str,
    threshold: float,
) -> bool:
    if operator == ">=":
        return actual_value >= threshold

    if operator == "<=":
        return actual_value <= threshold

    if operator == ">":
        return actual_value > threshold

    if operator == "<":
        return actual_value < threshold

    if operator == "==":
        return actual_value == threshold

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported quality gate operator: {operator}",
    )