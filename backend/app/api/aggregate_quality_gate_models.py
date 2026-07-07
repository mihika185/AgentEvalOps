from typing import Any, Optional

from pydantic import BaseModel

from backend.app.evaluation.experiment_quality_gates import AggregateQualityGateSummary


class AggregateQualityGateCheckResponse(BaseModel):
    gate_id: str
    gate_name: str
    metric_name: str
    metric_value: Optional[float]
    operator: str
    threshold: float
    passed: bool
    failure_reason: Optional[str]


class AggregateQualityGateSummaryResponse(BaseModel):
    scope_type: str
    scope_id: str
    profile_name: str
    evaluator_type: str
    overall_passed: bool
    passed_count: int
    failed_count: int
    total_gates: int
    pass_rate: float
    aggregate_metrics: dict[str, Any]
    checks: list[AggregateQualityGateCheckResponse]


def to_aggregate_quality_gate_summary_response(
    summary: AggregateQualityGateSummary,
) -> AggregateQualityGateSummaryResponse:
    return AggregateQualityGateSummaryResponse(
        scope_type=summary.scope_type,
        scope_id=summary.scope_id,
        profile_name=summary.profile_name,
        evaluator_type=summary.evaluator_type,
        overall_passed=summary.overall_passed,
        passed_count=summary.passed_count,
        failed_count=summary.failed_count,
        total_gates=summary.total_gates,
        pass_rate=summary.pass_rate,
        aggregate_metrics=summary.aggregate_metrics,
        checks=[
            AggregateQualityGateCheckResponse(
                gate_id=check.gate_id,
                gate_name=check.gate_name,
                metric_name=check.metric_name,
                metric_value=check.metric_value,
                operator=check.operator,
                threshold=check.threshold,
                passed=check.passed,
                failure_reason=check.failure_reason,
            )
            for check in summary.checks
        ],
    )