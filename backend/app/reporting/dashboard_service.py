from datetime import datetime, timezone
from typing import Any, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import (
    BenchmarkRun,
    Document,
    EvaluationResult,
    Experiment,
    Run,
)


DASHBOARD_SERVICE_VERSION = "dashboard-summary-v1"

def build_dashboard_summary(
    db: Session,
    recent_limit: int = 8,
) -> dict[str, Any]:
    runs = db.execute(
        select(Run)
        .order_by(Run.created_at.desc())
    ).scalars().all()

    recent_runs = runs[:recent_limit]

    experiments = db.execute(
        select(Experiment)
        .order_by(Experiment.created_at.desc())
        .limit(recent_limit)
    ).scalars().all()

    benchmark_runs = db.execute(
        select(BenchmarkRun)
        .order_by(BenchmarkRun.started_at.desc())
        .limit(recent_limit)
    ).scalars().all()

    recent_evaluation_results = db.execute(
        select(EvaluationResult)
        .order_by(EvaluationResult.created_at.desc())
        .limit(1000)
    ).scalars().all()

    return {
        "service_version": DASHBOARD_SERVICE_VERSION,
        "generated_at": utc_iso(),
        "counts": build_resource_counts(db),
        "run_health": build_run_health(runs),
        "latency_cost": build_latency_cost_summary(runs),
        "quality": build_quality_summary(recent_evaluation_results),
        "recent_runs": [
            run_to_dashboard_item(run)
            for run in recent_runs
        ],
        "recent_experiments": [
            experiment_to_dashboard_item(experiment)
            for experiment in experiments
        ],
        "recent_benchmark_runs": [
            benchmark_run_to_dashboard_item(benchmark_run)
            for benchmark_run in benchmark_runs
        ],
    }

def build_resource_counts(db: Session) -> dict[str, int]:
    return {
        "documents": count_rows(db, Document),
        "experiments": count_rows(db, Experiment),
        "runs": count_rows(db, Run),
        "benchmark_runs": count_rows(db, BenchmarkRun),
    }

def count_rows(db: Session, model) -> int:
    return db.query(model).count()

def build_run_health(runs: list[Run]) -> dict[str, Any]:
    total_runs = len(runs)
    completed_runs = count_runs_by_status(runs, "completed")
    failed_runs = count_runs_by_status(runs, "failed")
    running_runs = count_runs_by_status(runs, "running")

    return {
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "running_runs": running_runs,
        "completed_run_rate": ratio(completed_runs, total_runs),
        "failed_run_rate": ratio(failed_runs, total_runs),
    }

def build_latency_cost_summary(runs: list[Run]) -> dict[str, Any]:
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
        "average_latency_ms": average(latency_values),
        "max_latency_ms": max(latency_values) if latency_values else None,
        "average_prompt_tokens": average(prompt_token_values) or 0.0,
        "average_completion_tokens": average(completion_token_values) or 0.0,
        "average_total_tokens": average(total_token_values) or 0.0,
        "average_estimated_cost": average(estimated_cost_values) or 0.0,
        "total_estimated_cost": round(sum(estimated_cost_values), 8),
    }

def build_quality_summary(
    evaluation_results: list[EvaluationResult],
) -> dict[str, Any]:
    metric_groups = group_metric_values(evaluation_results)

    return {
        "average_overall_quality_score": average(
            metric_groups.get("overall_quality_score", [])
        ),
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
        "average_quality_gate_pass_rate": average(
            metric_groups.get("quality_gate_pass_rate", [])
        ),
        "quality_gate_overall_pass_rate": average(
            metric_groups.get("quality_gate_overall_pass", [])
        ),
        "metric_result_count": len(evaluation_results),
    }

def group_metric_values(
    evaluation_results: list[EvaluationResult],
) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {}

    for result in evaluation_results:
        groups.setdefault(result.metric_name, []).append(
            float(result.metric_value)
        )

    return groups

def run_to_dashboard_item(run: Run) -> dict[str, Any]:
    metadata = run.metadata_json or {}

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
        "quality_gate_passed": metadata.get("quality_gate_passed"),
        "response_blocked_by_quality_gate": metadata.get(
            "response_blocked_by_quality_gate"
        ),
        "retrieval_provider": metadata.get("retrieval_provider"),
        "answer_generator": metadata.get("answer_generator"),
        "created_at": run.created_at,
        "completed_at": run.completed_at,
    }

def experiment_to_dashboard_item(experiment: Experiment) -> dict[str, Any]:
    metadata = experiment.metadata_json or {}
    latest_gate_result = metadata.get("latest_experiment_quality_gate_result")

    return {
        "id": experiment.id,
        "name": experiment.name,
        "description": experiment.description,
        "retriever_type": experiment.retriever_type,
        "llm_provider": experiment.llm_provider,
        "llm_model": experiment.llm_model,
        "prompt_version": experiment.prompt_version,
        "chunking_strategy": experiment.chunking_strategy,
        "reranker_enabled": experiment.reranker_enabled,
        "latest_quality_gate_result": latest_gate_result,
        "created_at": experiment.created_at,
    }


def benchmark_run_to_dashboard_item(
    benchmark_run: BenchmarkRun,
) -> dict[str, Any]:
    pass_rate = ratio(
        benchmark_run.passed_cases,
        benchmark_run.total_cases,
    )

    metadata = benchmark_run.metadata_json or {}

    return {
        "id": benchmark_run.id,
        "dataset_id": benchmark_run.dataset_id,
        "status": benchmark_run.status,
        "total_cases": benchmark_run.total_cases,
        "passed_cases": benchmark_run.passed_cases,
        "failed_cases": benchmark_run.failed_cases,
        "pass_rate": pass_rate,
        "average_overall_quality_score": benchmark_run.average_overall_quality_score,
        "average_hallucination_risk": benchmark_run.average_hallucination_risk,
        "average_latency_ms": benchmark_run.average_latency_ms,
        "pipeline_config_id": metadata.get("pipeline_config_id"),
        "pipeline_config_name": metadata.get("pipeline_config_name"),
        "latest_quality_gate_result": metadata.get(
            "latest_benchmark_quality_gate_result"
        ),
        "started_at": benchmark_run.started_at,
        "completed_at": benchmark_run.completed_at,
    }


def count_runs_by_status(runs: list[Run], status: str) -> int:
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


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()