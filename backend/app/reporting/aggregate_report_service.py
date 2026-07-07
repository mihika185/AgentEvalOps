from datetime import datetime, timezone
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
from backend.app.evaluation.experiment_quality_gates import (
    AggregateQualityGateError,
    evaluate_benchmark_run_quality_gates,
    evaluate_experiment_quality_gates,
    summary_to_dict,
)


REPORT_SERVICE_VERSION = "aggregate-report-v1"


class AggregateReportError(Exception):
    pass


def build_experiment_report(
    db: Session,
    experiment_id: str,
    profile_name: str = "default-v1",
    recent_run_limit: int = 10,
) -> dict[str, Any]:
    experiment = db.get(Experiment, experiment_id)

    if experiment is None:
        raise AggregateReportError(
            f"Experiment with id '{experiment_id}' was not found"
        )

    runs = db.execute(
        select(Run)
        .where(Run.experiment_id == experiment_id)
        .order_by(Run.created_at.desc())
    ).scalars().all()

    evaluation_results = fetch_evaluation_results_for_runs(
        db=db,
        run_ids=[
            run.id
            for run in runs
        ],
    )

    try:
        gate_summary = evaluate_experiment_quality_gates(
            db=db,
            experiment_id=experiment_id,
            profile_name=profile_name,
            persist=True,
        )
    except AggregateQualityGateError as exc:
        raise AggregateReportError(str(exc)) from exc

    run_summary = summarize_runs(runs)
    metric_summaries = summarize_evaluation_results(evaluation_results)

    readiness_decision = build_readiness_decision(
        scope_type="experiment",
        gate_overall_passed=gate_summary.overall_passed,
        failed_count=int(run_summary["failed_runs"]),
        failure_rate=float(run_summary["failed_run_rate"]),
        average_hallucination_risk=number_or_none(
            gate_summary.aggregate_metrics.get("average_hallucination_risk")
        ),
    )

    return {
        "report_type": "experiment",
        "scope_id": experiment_id,
        "generated_at": utc_iso(),
        "service_version": REPORT_SERVICE_VERSION,
        "summary": {
            "experiment": {
                "id": experiment.id,
                "name": experiment.name,
                "description": experiment.description,
                "retriever_type": experiment.retriever_type,
                "llm_provider": experiment.llm_provider,
                "llm_model": experiment.llm_model,
                "prompt_version": experiment.prompt_version,
                "chunking_strategy": experiment.chunking_strategy,
                "reranker_enabled": experiment.reranker_enabled,
                "created_at": experiment.created_at,
            },
            "runs": run_summary,
            "metrics": metric_summaries,
        },
        "quality_gate_result": summary_to_dict(gate_summary),
        "readiness_decision": readiness_decision,
        "details": {
            "recent_runs": [
                run_to_report_item(run)
                for run in runs[:recent_run_limit]
            ],
            "metric_summary_count": len(metric_summaries),
            "report_inputs": {
                "profile_name": profile_name,
                "recent_run_limit": recent_run_limit,
            },
        },
    }


def build_benchmark_run_report(
    db: Session,
    benchmark_run_id: str,
    profile_name: str = "default-v1",
    failed_item_limit: int = 20,
) -> dict[str, Any]:
    benchmark_run = db.get(BenchmarkRun, benchmark_run_id)

    if benchmark_run is None:
        raise AggregateReportError(
            f"Benchmark run with id '{benchmark_run_id}' was not found"
        )

    run_items = db.execute(
        select(BenchmarkRunItem)
        .where(BenchmarkRunItem.benchmark_run_id == benchmark_run_id)
        .order_by(BenchmarkRunItem.created_at.asc())
    ).scalars().all()

    try:
        gate_summary = evaluate_benchmark_run_quality_gates(
            db=db,
            benchmark_run_id=benchmark_run_id,
            profile_name=profile_name,
            persist=True,
        )
    except AggregateQualityGateError as exc:
        raise AggregateReportError(str(exc)) from exc

    failed_items = [
        item
        for item in run_items
        if not item.passed
    ]

    failure_categories = summarize_failure_categories(failed_items)

    readiness_decision = build_readiness_decision(
        scope_type="benchmark_run",
        gate_overall_passed=gate_summary.overall_passed,
        failed_count=len(failed_items),
        failure_rate=ratio(len(failed_items), len(run_items)),
        average_hallucination_risk=number_or_none(
            gate_summary.aggregate_metrics.get("average_hallucination_risk")
        ),
    )

    return {
        "report_type": "benchmark_run",
        "scope_id": benchmark_run_id,
        "generated_at": utc_iso(),
        "service_version": REPORT_SERVICE_VERSION,
        "summary": {
            "benchmark_run": benchmark_run_to_summary(benchmark_run),
            "cases": summarize_benchmark_items(run_items),
            "failure_categories": failure_categories,
        },
        "quality_gate_result": summary_to_dict(gate_summary),
        "readiness_decision": readiness_decision,
        "details": {
            "failed_items": [
                benchmark_item_to_failure_report(item)
                for item in failed_items[:failed_item_limit]
            ],
            "report_inputs": {
                "profile_name": profile_name,
                "failed_item_limit": failed_item_limit,
            },
        },
    }


def fetch_evaluation_results_for_runs(
    db: Session,
    run_ids: list[str],
) -> list[EvaluationResult]:
    if not run_ids:
        return []

    return db.execute(
        select(EvaluationResult)
        .where(EvaluationResult.run_id.in_(run_ids))
    ).scalars().all()


def summarize_runs(runs: list[Run]) -> dict[str, Any]:
    total_runs = len(runs)
    completed_runs = count_by_status(runs, "completed")
    failed_runs = count_by_status(runs, "failed")
    running_runs = count_by_status(runs, "running")

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

    estimated_cost_values = [
        float(run.estimated_cost)
        for run in runs
        if run.estimated_cost is not None
    ]

    return {
        "total_runs": float(total_runs),
        "completed_runs": float(completed_runs),
        "failed_runs": float(failed_runs),
        "running_runs": float(running_runs),
        "completed_run_rate": ratio(completed_runs, total_runs),
        "failed_run_rate": ratio(failed_runs, total_runs),
        "average_latency_ms": average(latency_values),
        "average_prompt_tokens": average(prompt_token_values) or 0.0,
        "average_completion_tokens": average(completion_token_values) or 0.0,
        "average_total_tokens": average(total_token_values) or 0.0,
        "average_estimated_cost": average(estimated_cost_values) or 0.0,
        "total_estimated_cost": round(sum(estimated_cost_values), 8),
    }


def summarize_evaluation_results(
    evaluation_results: list[EvaluationResult],
) -> list[dict[str, Any]]:
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
            {
                "evaluator_type": evaluator_type,
                "metric_name": metric_name,
                "count": len(values),
                "average_value": round(sum(values) / len(values), 4),
                "min_value": round(min(values), 4),
                "max_value": round(max(values), 4),
            }
        )

    return sorted(
        summaries,
        key=lambda item: (
            item["evaluator_type"],
            item["metric_name"],
        ),
    )


def summarize_benchmark_items(
    run_items: list[BenchmarkRunItem],
) -> dict[str, Any]:
    total_cases = len(run_items)
    passed_cases = len([
        item
        for item in run_items
        if item.passed
    ])
    failed_cases = total_cases - passed_cases

    answerable_cases = len([
        item
        for item in run_items
        if item.expected_behavior == "answerable"
    ])

    unanswerable_cases = len([
        item
        for item in run_items
        if item.expected_behavior == "unanswerable"
    ])

    latency_values = [
        float(item.latency_ms)
        for item in run_items
        if item.latency_ms is not None
    ]

    return {
        "total_cases": float(total_cases),
        "passed_cases": float(passed_cases),
        "failed_cases": float(failed_cases),
        "pass_rate": ratio(passed_cases, total_cases),
        "failed_case_rate": ratio(failed_cases, total_cases),
        "answerable_cases": float(answerable_cases),
        "unanswerable_cases": float(unanswerable_cases),
        "average_latency_ms": average(latency_values),
    }


def summarize_failure_categories(
    failed_items: list[BenchmarkRunItem],
) -> dict[str, int]:
    categories: dict[str, int] = {}

    for item in failed_items:
        category = categorize_benchmark_failure(item)

        categories[category] = categories.get(category, 0) + 1

    return dict(
        sorted(
            categories.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def build_readiness_decision(
    scope_type: str,
    gate_overall_passed: bool,
    failed_count: int,
    failure_rate: float,
    average_hallucination_risk: Optional[float],
) -> dict[str, Any]:
    reasons: list[str] = []

    if not gate_overall_passed:
        reasons.append("aggregate_quality_gates_failed")

    if failed_count > 0:
        reasons.append("failures_present")

    if failure_rate > 0:
        reasons.append("failure_rate_above_zero")

    if (
        average_hallucination_risk is not None
        and average_hallucination_risk > 0.20
    ):
        reasons.append("hallucination_risk_above_release_bar")

    ready = not reasons

    return {
        "ready": ready,
        "status": "ready" if ready else "needs_attention",
        "scope_type": scope_type,
        "reasons": reasons if reasons else ["all_release_checks_passed"],
        "recommendation": (
            "Ready to proceed to frontend/demo usage."
            if ready
            else "Review failed gates, failures, latency, and hallucination metrics before release."
        ),
    }


def run_to_report_item(run: Run) -> dict[str, Any]:
    return {
        "id": run.id,
        "experiment_id": run.experiment_id,
        "workflow_type": run.workflow_type,
        "status": run.status,
        "input_query": run.input_query,
        "latency_ms": run.latency_ms,
        "prompt_tokens": run.prompt_tokens,
        "completion_tokens": run.completion_tokens,
        "estimated_cost": run.estimated_cost,
        "created_at": run.created_at,
        "completed_at": run.completed_at,
        "quality_gate_passed": (run.metadata_json or {}).get("quality_gate_passed"),
        "response_blocked_by_quality_gate": (
            run.metadata_json or {}
        ).get("response_blocked_by_quality_gate"),
    }


def benchmark_run_to_summary(
    benchmark_run: BenchmarkRun,
) -> dict[str, Any]:
    pass_rate = ratio(
        benchmark_run.passed_cases,
        benchmark_run.total_cases,
    )

    failed_case_rate = ratio(
        benchmark_run.failed_cases,
        benchmark_run.total_cases,
    )

    return {
        "id": benchmark_run.id,
        "dataset_id": benchmark_run.dataset_id,
        "status": benchmark_run.status,
        "total_cases": float(benchmark_run.total_cases),
        "passed_cases": float(benchmark_run.passed_cases),
        "failed_cases": float(benchmark_run.failed_cases),
        "pass_rate": pass_rate,
        "failed_case_rate": failed_case_rate,
        "answerable_cases": float(benchmark_run.answerable_cases),
        "answerable_passed": float(benchmark_run.answerable_passed),
        "unanswerable_cases": float(benchmark_run.unanswerable_cases),
        "unanswerable_passed": float(benchmark_run.unanswerable_passed),
        "average_answer_support_score": benchmark_run.average_answer_support_score,
        "average_query_answer_relevance_score": (
            benchmark_run.average_query_answer_relevance_score
        ),
        "average_hallucination_risk": benchmark_run.average_hallucination_risk,
        "average_overall_quality_score": benchmark_run.average_overall_quality_score,
        "average_latency_ms": benchmark_run.average_latency_ms,
        "metadata_json": benchmark_run.metadata_json or {},
        "started_at": benchmark_run.started_at,
        "completed_at": benchmark_run.completed_at,
    }


def benchmark_item_to_failure_report(
    item: BenchmarkRunItem,
) -> dict[str, Any]:
    return {
        "id": item.id,
        "test_case_id": item.test_case_id,
        "rag_run_id": item.rag_run_id,
        "question": item.question,
        "expected_behavior": item.expected_behavior,
        "expected_keywords": item.expected_keywords,
        "actual_answer": item.actual_answer,
        "failure_reason": item.failure_reason,
        "failure_category": categorize_benchmark_failure(item),
        "quality_gate_passed": item.quality_gate_passed,
        "response_blocked_by_quality_gate": item.response_blocked_by_quality_gate,
        "metrics_json": item.metrics_json or {},
        "metadata_json": item.metadata_json or {},
    }


def categorize_benchmark_failure(item: BenchmarkRunItem) -> str:
    if item.passed:
        return "passed"

    failure_reason = (item.failure_reason or "").lower()

    if "missing expected keywords" in failure_reason:
        return "missing_expected_keywords"

    if "quality gate blocked" in failure_reason:
        return "answer_blocked_by_quality_gate"

    if "quality gates did not pass" in failure_reason:
        return "quality_gate_failed"

    if "unanswerable query to be blocked" in failure_reason:
        return "unanswerable_answer_returned"

    if (item.metadata_json or {}).get("error_type") == "RAGWorkflowError":
        return "rag_workflow_error"

    return "unknown_failure"


def count_by_status(runs: list[Run], status: str) -> int:
    return sum(
        1
        for run in runs
        if run.status == status
    )


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


def number_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()