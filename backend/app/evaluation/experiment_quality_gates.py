from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import (
    BenchmarkRun,
    BenchmarkRunItem,
    EvaluationResult,
    Experiment,
    Run,
)


AGGREGATE_QUALITY_GATE_EVALUATOR_TYPE = "aggregate-quality-gate-evaluator-v1"
DEFAULT_AGGREGATE_QUALITY_GATE_PROFILE = "default-v1"


class AggregateQualityGateError(Exception):
    pass


@dataclass(frozen=True)
class AggregateQualityGate:
    name: str
    metric_name: str
    operator: str
    threshold: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AggregateQualityGateCheck:
    gate_id: str
    gate_name: str
    metric_name: str
    metric_value: Optional[float]
    operator: str
    threshold: float
    passed: bool
    failure_reason: Optional[str]


@dataclass(frozen=True)
class AggregateQualityGateSummary:
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
    checks: list[AggregateQualityGateCheck]


EXPERIMENT_QUALITY_GATE_PROFILES: dict[str, list[AggregateQualityGate]] = {
    "default-v1": [
        AggregateQualityGate(
            name="Minimum Completed Run Rate",
            metric_name="completed_run_rate",
            operator=">=",
            threshold=0.90,
            metadata={
                "description": "Most experiment runs should complete successfully."
            },
        ),
        AggregateQualityGate(
            name="Maximum Failed Run Rate",
            metric_name="failed_run_rate",
            operator="<=",
            threshold=0.10,
            metadata={
                "description": "Failed workflow runs should stay low."
            },
        ),
        AggregateQualityGate(
            name="Minimum Average Overall Quality",
            metric_name="average_overall_quality_score",
            operator=">=",
            threshold=0.70,
            metadata={
                "description": "Average RAG quality should pass the minimum release bar."
            },
        ),
        AggregateQualityGate(
            name="Maximum Average Hallucination Risk",
            metric_name="average_hallucination_risk",
            operator="<=",
            threshold=0.20,
            metadata={
                "description": "Average hallucination risk should remain low."
            },
        ),
        AggregateQualityGate(
            name="Maximum Average Latency",
            metric_name="average_latency_ms",
            operator="<=",
            threshold=15000.0,
            metadata={
                "description": "Average latency should stay within an acceptable local-development limit."
            },
        ),
        AggregateQualityGate(
            name="Maximum Average Estimated Cost",
            metric_name="average_estimated_cost",
            operator="<=",
            threshold=0.05,
            metadata={
                "description": "Average answer-generation cost should remain controlled."
            },
        ),
    ],
    "strict-v1": [
        AggregateQualityGate(
            name="Minimum Completed Run Rate",
            metric_name="completed_run_rate",
            operator=">=",
            threshold=0.98,
            metadata={},
        ),
        AggregateQualityGate(
            name="Maximum Failed Run Rate",
            metric_name="failed_run_rate",
            operator="<=",
            threshold=0.02,
            metadata={},
        ),
        AggregateQualityGate(
            name="Minimum Average Overall Quality",
            metric_name="average_overall_quality_score",
            operator=">=",
            threshold=0.85,
            metadata={},
        ),
        AggregateQualityGate(
            name="Maximum Average Hallucination Risk",
            metric_name="average_hallucination_risk",
            operator="<=",
            threshold=0.10,
            metadata={},
        ),
        AggregateQualityGate(
            name="Maximum Average Latency",
            metric_name="average_latency_ms",
            operator="<=",
            threshold=8000.0,
            metadata={},
        ),
        AggregateQualityGate(
            name="Maximum Average Estimated Cost",
            metric_name="average_estimated_cost",
            operator="<=",
            threshold=0.02,
            metadata={},
        ),
    ],
}


BENCHMARK_QUALITY_GATE_PROFILES: dict[str, list[AggregateQualityGate]] = {
    "default-v1": [
        AggregateQualityGate(
            name="Minimum Benchmark Pass Rate",
            metric_name="pass_rate",
            operator=">=",
            threshold=0.80,
            metadata={
                "description": "Benchmark run should pass most test cases."
            },
        ),
        AggregateQualityGate(
            name="Maximum Failed Case Rate",
            metric_name="failed_case_rate",
            operator="<=",
            threshold=0.20,
            metadata={
                "description": "Failed benchmark cases should stay under the release limit."
            },
        ),
        AggregateQualityGate(
            name="Minimum Average Overall Quality",
            metric_name="average_overall_quality_score",
            operator=">=",
            threshold=0.70,
            metadata={},
        ),
        AggregateQualityGate(
            name="Maximum Average Hallucination Risk",
            metric_name="average_hallucination_risk",
            operator="<=",
            threshold=0.20,
            metadata={},
        ),
        AggregateQualityGate(
            name="Maximum Average Latency",
            metric_name="average_latency_ms",
            operator="<=",
            threshold=15000.0,
            metadata={},
        ),
        AggregateQualityGate(
            name="Maximum Average Estimated Cost",
            metric_name="average_estimated_cost",
            operator="<=",
            threshold=0.05,
            metadata={},
        ),
    ],
    "strict-v1": [
        AggregateQualityGate(
            name="Minimum Benchmark Pass Rate",
            metric_name="pass_rate",
            operator=">=",
            threshold=0.95,
            metadata={},
        ),
        AggregateQualityGate(
            name="Maximum Failed Case Rate",
            metric_name="failed_case_rate",
            operator="<=",
            threshold=0.05,
            metadata={},
        ),
        AggregateQualityGate(
            name="Minimum Average Overall Quality",
            metric_name="average_overall_quality_score",
            operator=">=",
            threshold=0.85,
            metadata={},
        ),
        AggregateQualityGate(
            name="Maximum Average Hallucination Risk",
            metric_name="average_hallucination_risk",
            operator="<=",
            threshold=0.10,
            metadata={},
        ),
        AggregateQualityGate(
            name="Maximum Average Latency",
            metric_name="average_latency_ms",
            operator="<=",
            threshold=8000.0,
            metadata={},
        ),
        AggregateQualityGate(
            name="Maximum Average Estimated Cost",
            metric_name="average_estimated_cost",
            operator="<=",
            threshold=0.02,
            metadata={},
        ),
    ],
}


def evaluate_experiment_quality_gates(
    db: Session,
    experiment_id: str,
    profile_name: str = DEFAULT_AGGREGATE_QUALITY_GATE_PROFILE,
    persist: bool = True,
) -> AggregateQualityGateSummary:
    experiment = db.get(Experiment, experiment_id)

    if experiment is None:
        raise AggregateQualityGateError(
            f"Experiment with id '{experiment_id}' was not found"
        )

    runs = db.execute(
        select(Run)
        .where(Run.experiment_id == experiment_id)
        .order_by(Run.created_at.asc())
    ).scalars().all()

    aggregate_metrics = build_experiment_aggregate_metrics(
        db=db,
        runs=runs,
    )

    resolved_profile_name = normalize_profile_name(
        profile_name=profile_name,
        profiles=EXPERIMENT_QUALITY_GATE_PROFILES,
    )

    summary = evaluate_aggregate_quality_gates(
        scope_type="experiment",
        scope_id=experiment_id,
        profile_name=resolved_profile_name,
        aggregate_metrics=aggregate_metrics,
        gates=EXPERIMENT_QUALITY_GATE_PROFILES[resolved_profile_name],
    )

    if persist:
        experiment.metadata_json = {
            **(experiment.metadata_json or {}),
            "latest_experiment_quality_gate_result": summary_to_dict(summary),
        }

        db.commit()
        db.refresh(experiment)

    return summary


def evaluate_benchmark_run_quality_gates(
    db: Session,
    benchmark_run_id: str,
    profile_name: str = DEFAULT_AGGREGATE_QUALITY_GATE_PROFILE,
    persist: bool = True,
) -> AggregateQualityGateSummary:
    benchmark_run = db.get(BenchmarkRun, benchmark_run_id)

    if benchmark_run is None:
        raise AggregateQualityGateError(
            f"Benchmark run with id '{benchmark_run_id}' was not found"
        )

    run_items = db.execute(
        select(BenchmarkRunItem)
        .where(BenchmarkRunItem.benchmark_run_id == benchmark_run_id)
        .order_by(BenchmarkRunItem.created_at.asc())
    ).scalars().all()

    aggregate_metrics = build_benchmark_run_aggregate_metrics(
        benchmark_run=benchmark_run,
        run_items=run_items,
    )

    resolved_profile_name = normalize_profile_name(
        profile_name=profile_name,
        profiles=BENCHMARK_QUALITY_GATE_PROFILES,
    )

    summary = evaluate_aggregate_quality_gates(
        scope_type="benchmark_run",
        scope_id=benchmark_run_id,
        profile_name=resolved_profile_name,
        aggregate_metrics=aggregate_metrics,
        gates=BENCHMARK_QUALITY_GATE_PROFILES[resolved_profile_name],
    )

    if persist:
        benchmark_run.metadata_json = {
            **(benchmark_run.metadata_json or {}),
            "latest_benchmark_quality_gate_result": summary_to_dict(summary),
        }

        db.commit()
        db.refresh(benchmark_run)

    return summary


def build_experiment_aggregate_metrics(
    db: Session,
    runs: list[Run],
) -> dict[str, Any]:
    total_runs = len(runs)
    completed_runs = count_runs_with_status(runs, "completed")
    failed_runs = count_runs_with_status(runs, "failed")

    run_ids = [
        run.id
        for run in runs
    ]

    evaluation_results = []

    if run_ids:
        evaluation_results = db.execute(
            select(EvaluationResult)
            .where(EvaluationResult.run_id.in_(run_ids))
        ).scalars().all()

    metric_groups = group_evaluation_metric_values(evaluation_results)

    latency_values = [
        float(run.latency_ms)
        for run in runs
        if run.latency_ms is not None
    ]

    prompt_token_values = [
        float(run.prompt_tokens)
        for run in runs
        if run.prompt_tokens is not None
    ]

    completion_token_values = [
        float(run.completion_tokens)
        for run in runs
        if run.completion_tokens is not None
    ]

    estimated_cost_values = [
        float(run.estimated_cost)
        for run in runs
        if run.estimated_cost is not None
    ]

    total_token_values = []

    for run in runs:
        total_tokens = read_number(run.metadata_json or {}, "total_tokens")

        if total_tokens is not None:
            total_token_values.append(total_tokens)
            continue

        if run.prompt_tokens is not None and run.completion_tokens is not None:
            total_token_values.append(
                float(run.prompt_tokens + run.completion_tokens)
            )

    return {
        "total_runs": float(total_runs),
        "completed_runs": float(completed_runs),
        "failed_runs": float(failed_runs),
        "completed_run_rate": ratio(completed_runs, total_runs),
        "failed_run_rate": ratio(failed_runs, total_runs),
        "average_latency_ms": average(latency_values),
        "average_prompt_tokens": average(prompt_token_values) or 0.0,
        "average_completion_tokens": average(completion_token_values) or 0.0,
        "average_total_tokens": average(total_token_values) or 0.0,
        "average_estimated_cost": average(estimated_cost_values) or 0.0,
        "total_estimated_cost": round(sum(estimated_cost_values), 8),
        "average_answer_support_score": average(
            metric_groups.get("answer_support_score", [])
        ),
        "average_query_answer_relevance_score": average(
            metric_groups.get("query_answer_relevance_score", [])
        ),
        "average_faithfulness_score": average(
            metric_groups.get("faithfulness_score", [])
        ),
        "average_hallucination_risk": average(
            metric_groups.get("hallucination_risk", [])
        ),
        "average_hallucination_rate": average(
            metric_groups.get("hallucination_rate", [])
        ),
        "average_citation_accuracy_score": average(
            metric_groups.get("citation_accuracy_score", [])
        ),
        "average_overall_quality_score": average(
            metric_groups.get("overall_quality_score", [])
        ),
        "evaluation_result_count": float(len(evaluation_results)),
    }


def build_benchmark_run_aggregate_metrics(
    benchmark_run: BenchmarkRun,
    run_items: list[BenchmarkRunItem],
) -> dict[str, Any]:
    total_cases = benchmark_run.total_cases or len(run_items)
    passed_cases = benchmark_run.passed_cases
    failed_cases = benchmark_run.failed_cases

    latency_values = [
        float(item.latency_ms)
        for item in run_items
        if item.latency_ms is not None
    ]

    estimated_cost_values = []
    prompt_token_values = []
    completion_token_values = []
    total_token_values = []

    for item in run_items:
        metrics = item.metrics_json or {}
        metadata = item.metadata_json or {}

        estimated_cost = read_number(metrics, "estimated_cost")

        if estimated_cost is None:
            estimated_cost = read_number(metadata, "estimated_cost")

        if estimated_cost is not None:
            estimated_cost_values.append(estimated_cost)

        prompt_tokens = read_number(metrics, "prompt_tokens")

        if prompt_tokens is None:
            prompt_tokens = read_number(metadata, "prompt_tokens")

        if prompt_tokens is not None:
            prompt_token_values.append(prompt_tokens)

        completion_tokens = read_number(metrics, "completion_tokens")

        if completion_tokens is None:
            completion_tokens = read_number(metadata, "completion_tokens")

        if completion_tokens is not None:
            completion_token_values.append(completion_tokens)

        total_tokens = read_number(metrics, "total_tokens")

        if total_tokens is None:
            total_tokens = read_number(metadata, "total_tokens")

        if total_tokens is not None:
            total_token_values.append(total_tokens)

    answerable_accuracy = ratio(
        benchmark_run.answerable_passed,
        benchmark_run.answerable_cases,
    )

    unanswerable_accuracy = ratio(
        benchmark_run.unanswerable_passed,
        benchmark_run.unanswerable_cases,
    )

    return {
        "total_cases": float(total_cases),
        "passed_cases": float(passed_cases),
        "failed_cases": float(failed_cases),
        "pass_rate": ratio(passed_cases, total_cases),
        "failed_case_rate": ratio(failed_cases, total_cases),
        "answerable_accuracy": answerable_accuracy,
        "unanswerable_accuracy": unanswerable_accuracy,
        "average_answer_support_score": benchmark_run.average_answer_support_score,
        "average_query_answer_relevance_score": benchmark_run.average_query_answer_relevance_score,
        "average_hallucination_risk": benchmark_run.average_hallucination_risk,
        "average_overall_quality_score": benchmark_run.average_overall_quality_score,
        "average_latency_ms": benchmark_run.average_latency_ms or average(latency_values),
        "average_prompt_tokens": average(prompt_token_values) or 0.0,
        "average_completion_tokens": average(completion_token_values) or 0.0,
        "average_total_tokens": average(total_token_values) or 0.0,
        "average_estimated_cost": average(estimated_cost_values) or 0.0,
        "total_estimated_cost": round(sum(estimated_cost_values), 8),
    }


def evaluate_aggregate_quality_gates(
    scope_type: str,
    scope_id: str,
    profile_name: str,
    aggregate_metrics: dict[str, Any],
    gates: list[AggregateQualityGate],
) -> AggregateQualityGateSummary:
    checks: list[AggregateQualityGateCheck] = []

    for gate in gates:
        metric_value = read_number(aggregate_metrics, gate.metric_name)

        if metric_value is None:
            checks.append(
                AggregateQualityGateCheck(
                    gate_id=gate_id(scope_type, profile_name, gate.name),
                    gate_name=gate.name,
                    metric_name=gate.metric_name,
                    metric_value=None,
                    operator=gate.operator,
                    threshold=gate.threshold,
                    passed=False,
                    failure_reason="metric_missing",
                )
            )
            continue

        passed = compare_metric(
            metric_value=metric_value,
            operator=gate.operator,
            threshold=gate.threshold,
        )

        checks.append(
            AggregateQualityGateCheck(
                gate_id=gate_id(scope_type, profile_name, gate.name),
                gate_name=gate.name,
                metric_name=gate.metric_name,
                metric_value=round(metric_value, 4),
                operator=gate.operator,
                threshold=gate.threshold,
                passed=passed,
                failure_reason=None if passed else "threshold_not_met",
            )
        )

    passed_count = sum(1 for check in checks if check.passed)
    failed_count = len(checks) - passed_count
    total_gates = len(checks)

    return AggregateQualityGateSummary(
        scope_type=scope_type,
        scope_id=scope_id,
        profile_name=profile_name,
        evaluator_type=AGGREGATE_QUALITY_GATE_EVALUATOR_TYPE,
        overall_passed=failed_count == 0,
        passed_count=passed_count,
        failed_count=failed_count,
        total_gates=total_gates,
        pass_rate=round(ratio(passed_count, total_gates), 4),
        aggregate_metrics=normalize_metrics_for_output(aggregate_metrics),
        checks=checks,
    )


def normalize_profile_name(
    profile_name: Optional[str],
    profiles: dict[str, list[AggregateQualityGate]],
) -> str:
    if profile_name is None:
        return DEFAULT_AGGREGATE_QUALITY_GATE_PROFILE

    cleaned_profile_name = profile_name.strip().lower()

    if not cleaned_profile_name:
        return DEFAULT_AGGREGATE_QUALITY_GATE_PROFILE

    if cleaned_profile_name not in profiles:
        supported_profiles = ", ".join(sorted(profiles.keys()))

        raise AggregateQualityGateError(
            f"Unsupported aggregate quality gate profile '{profile_name}'. "
            f"Supported profiles are: {supported_profiles}"
        )

    return cleaned_profile_name


def count_runs_with_status(runs: list[Run], status: str) -> int:
    return sum(
        1
        for run in runs
        if run.status == status
    )


def group_evaluation_metric_values(
    evaluation_results: list[EvaluationResult],
) -> dict[str, list[float]]:
    grouped_values: dict[str, list[float]] = {}

    for result in evaluation_results:
        grouped_values.setdefault(result.metric_name, []).append(
            float(result.metric_value)
        )

    return grouped_values


def average(values: list[float]) -> Optional[float]:
    if not values:
        return None

    return round(sum(values) / len(values), 4)


def ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0

    return round(float(numerator) / float(denominator), 4)


def read_number(data: dict[str, Any], key: str) -> Optional[float]:
    value = data.get(key)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_metric(
    metric_value: float,
    operator: str,
    threshold: float,
) -> bool:
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

    raise AggregateQualityGateError(
        f"Unsupported aggregate quality gate operator: {operator}"
    )


def gate_id(scope_type: str, profile_name: str, gate_name: str) -> str:
    normalized_gate_name = (
        gate_name
        .lower()
        .replace(" ", "-")
        .replace("_", "-")
    )

    return f"{scope_type}:{profile_name}:{normalized_gate_name}"


def normalize_metrics_for_output(metrics: dict[str, Any]) -> dict[str, Any]:
    normalized_metrics = {}

    for key, value in metrics.items():
        if isinstance(value, float):
            normalized_metrics[key] = round(value, 4)
        else:
            normalized_metrics[key] = value

    return normalized_metrics


def summary_to_dict(summary: AggregateQualityGateSummary) -> dict[str, Any]:
    return {
        "scope_type": summary.scope_type,
        "scope_id": summary.scope_id,
        "profile_name": summary.profile_name,
        "evaluator_type": summary.evaluator_type,
        "overall_passed": summary.overall_passed,
        "passed_count": summary.passed_count,
        "failed_count": summary.failed_count,
        "total_gates": summary.total_gates,
        "pass_rate": summary.pass_rate,
        "aggregate_metrics": summary.aggregate_metrics,
        "checks": [
            {
                "gate_id": check.gate_id,
                "gate_name": check.gate_name,
                "metric_name": check.metric_name,
                "metric_value": check.metric_value,
                "operator": check.operator,
                "threshold": check.threshold,
                "passed": check.passed,
                "failure_reason": check.failure_reason,
            }
            for check in summary.checks
        ],
    }