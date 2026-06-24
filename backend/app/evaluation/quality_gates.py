from dataclasses import dataclass
from typing import Any
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import EvaluationResult, QualityGate, Run
from backend.app.evaluation.answer_evaluator import (
    EVALUATOR_TYPE as RAG_EVALUATOR_TYPE,
    EvaluationError,
    evaluate_rag_run,
)
from backend.app.logging_config import get_logger


logger = get_logger(__name__)

QUALITY_GATE_EVALUATOR_TYPE = "quality-gate-evaluator-v1"


class QualityGateError(Exception):
    pass

@dataclass(frozen=True)
class DefaultQualityGate:
    name: str
    metric_name: str
    operator: str
    threshold: float
    metadata: dict[str, Any]

@dataclass(frozen=True)
class QualityGateCheck:
    gate_id: str
    gate_name: str
    metric_name: str
    metric_value: float
    operator: str
    threshold: float
    passed: bool


@dataclass(frozen=True)
class QualityGateSummary:
    run_id: str
    overall_passed: bool
    passed_count: int
    failed_count: int
    total_gates: int
    pass_rate: float
    checks: list[QualityGateCheck]

DEFAULT_QUALITY_GATES = [
    DefaultQualityGate(
        name="Minimum Answer Support",
        metric_name="answer_support_score",
        operator=">=",
        threshold=0.80,
        metadata={
            "description": "Answer should be mostly supported by retrieved context."
        }
    ),
    DefaultQualityGate(
        name="Maximum Hallucination Risk",
        metric_name="hallucination_risk",
        operator="<=",
        threshold=0.20,
        metadata={
            "description": "Unsupported answer terms should remain low."
        }
    ),
    DefaultQualityGate(
        name="Minimum Top Retrieval Score",
        metric_name="top_retrieval_score",
        operator=">=",
        threshold=0.35,
        metadata={
            "description": "At least one retrieved chunk should be strongly relevant."
        }
    ),
    DefaultQualityGate(
        name="Minimum Overall Quality",
        metric_name="overall_quality_score",
        operator=">=",
        threshold=0.70,
        metadata={
            "description": "Combined RAG quality score should pass minimum threshold."
        }
    ),
    DefaultQualityGate(
        name="Minimum Source Chunks",
        metric_name="source_chunk_count",
        operator=">=",
        threshold=1.0,
        metadata={
            "description": "Answer should have at least one retrieved source chunk."
        }
    ),
]

def ensure_default_quality_gates(db: Session) -> int:
    created_count = 0

    for gate in DEFAULT_QUALITY_GATES:
        existing_gate = db.execute(
            select(QualityGate).where(QualityGate.name == gate.name)
        ).scalars().first()

        if existing_gate is not None:
            continue

        db.add(
            QualityGate(
                name=gate.name,
                metric_name=gate.metric_name,
                operator=gate.operator,
                threshold=gate.threshold,
                is_active=True,
                metadata_json=gate.metadata
            )
        )

        created_count += 1

    db.commit()

    logger.info("Created %s default quality gates", created_count)
    return created_count

def evaluate_quality_gates(
    db: Session,
    run_id: str,
    persist: bool = True
) -> QualityGateSummary:
    run = db.get(Run, run_id)

    if run is None:
        raise QualityGateError(f"Run with id '{run_id}' was not found")

    if run.status != "completed":
        raise QualityGateError(
            f"Run '{run_id}' must be completed before quality gates can run"
        )

    ensure_default_quality_gates(db)

    metric_values = get_metric_values(db, run_id)

    if not metric_values:
        try:
            evaluate_rag_run(db=db, run_id=run_id, persist=True)
        except EvaluationError as exc:
            raise QualityGateError(str(exc)) from exc

        metric_values = get_metric_values(db, run_id)

    active_gates = db.execute(
        select(QualityGate)
        .where(QualityGate.is_active.is_(True))
        .order_by(QualityGate.created_at.asc())
    ).scalars().all()

    if not active_gates:
        raise QualityGateError("No active quality gates found")

    checks: list[QualityGateCheck] = []

    for gate in active_gates:
        if gate.metric_name not in metric_values:
            raise QualityGateError(
                f"Metric '{gate.metric_name}' required by gate "
                f"'{gate.name}' was not found for run '{run_id}'"
            )

        metric_value = metric_values[gate.metric_name]
        passed = compare_metric(
            metric_value=metric_value,
            operator=gate.operator,
            threshold=gate.threshold
        )

        checks.append(
            QualityGateCheck(
                gate_id=gate.id,
                gate_name=gate.name,
                metric_name=gate.metric_name,
                metric_value=metric_value,
                operator=gate.operator,
                threshold=gate.threshold,
                passed=passed
            )
        )

    passed_count = sum(1 for check in checks if check.passed)
    failed_count = len(checks) - passed_count
    overall_passed = failed_count == 0
    pass_rate = passed_count / len(checks)

    summary = QualityGateSummary(
        run_id=run_id,
        overall_passed=overall_passed,
        passed_count=passed_count,
        failed_count=failed_count,
        total_gates=len(checks),
        pass_rate=round(pass_rate, 4),
        checks=checks
    )

    if persist:
        save_quality_gate_summary(db, run, summary)

    logger.info(
        "Quality gates evaluated for run %s: %s/%s passed",
        run_id,
        passed_count,
        len(checks)
    )
    return summary

def get_metric_values(db: Session, run_id: str) -> dict[str, float]:
    results = db.execute(
        select(EvaluationResult)
        .where(
            EvaluationResult.run_id == run_id,
            EvaluationResult.evaluator_type == RAG_EVALUATOR_TYPE
        )
    ).scalars().all()

    return {
        result.metric_name: result.metric_value
        for result in results
    }

def compare_metric(metric_value: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return metric_value >= threshold
    if operator == ">":
        return metric_value > threshold
    if operator == "<=":
        return metric_value <= threshold
    if operator == "<":
        return metric_value < threshold
    if operator == "==":
        return metric_value == threshold
    if operator == "!=":
        return metric_value != threshold

    raise QualityGateError(f"Unsupported quality gate operator: {operator}")

def save_quality_gate_summary(
    db: Session,
    run: Run,
    summary: QualityGateSummary
) -> None:
    existing_results = db.execute(
        select(EvaluationResult)
        .where(
            EvaluationResult.run_id == summary.run_id,
            EvaluationResult.evaluator_type == QUALITY_GATE_EVALUATOR_TYPE
        )
    ).scalars().all()

    for result in existing_results:
        db.delete(result)

    check_details = [
        {
            "gate_id": check.gate_id,
            "gate_name": check.gate_name,
            "metric_name": check.metric_name,
            "metric_value": check.metric_value,
            "operator": check.operator,
            "threshold": check.threshold,
            "passed": check.passed
        }
        for check in summary.checks
    ]

    db.add(
        EvaluationResult(
            run_id=summary.run_id,
            metric_name="quality_gate_overall_pass",
            metric_value=1.0 if summary.overall_passed else 0.0,
            evaluator_type=QUALITY_GATE_EVALUATOR_TYPE,
            details_json={
                "checks": check_details
            }
        )
    )

    db.add(
        EvaluationResult(
            run_id=summary.run_id,
            metric_name="quality_gate_pass_rate",
            metric_value=summary.pass_rate,
            evaluator_type=QUALITY_GATE_EVALUATOR_TYPE,
            details_json={
                "passed_count": summary.passed_count,
                "failed_count": summary.failed_count,
                "total_gates": summary.total_gates
            }
        )
    )

    db.add(
        EvaluationResult(
            run_id=summary.run_id,
            metric_name="quality_gate_failed_count",
            metric_value=float(summary.failed_count),
            evaluator_type=QUALITY_GATE_EVALUATOR_TYPE,
            details_json={
                "failed_gates": [
                    check.gate_name
                    for check in summary.checks
                    if not check.passed
                ]
            }
        )
    )

    run.metadata_json = {
        **(run.metadata_json or {}),
        "quality_gate_overall_pass": summary.overall_passed,
        "quality_gate_pass_rate": summary.pass_rate,
        "quality_gate_failed_count": summary.failed_count
    }

    db.commit()