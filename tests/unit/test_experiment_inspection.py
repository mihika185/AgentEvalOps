from datetime import datetime, timezone
from types import SimpleNamespace

from backend.app.api.experiments import (
    average_latency_ms,
    build_experiment_inspection_summary,
    build_metric_summaries,
    count_by_field,
)


def test_count_by_field_counts_values():
    items = [
        SimpleNamespace(workflow_type="rag_answer"),
        SimpleNamespace(workflow_type="tool_calling_agent"),
        SimpleNamespace(workflow_type="rag_answer"),
    ]

    assert count_by_field(items, "workflow_type") == {
        "rag_answer": 2,
        "tool_calling_agent": 1,
    }

def test_average_latency_ms_ignores_missing_values():
    runs = [
        SimpleNamespace(latency_ms=100),
        SimpleNamespace(latency_ms=None),
        SimpleNamespace(latency_ms=200),
    ]

    assert average_latency_ms(runs) == 150.0

def test_build_metric_summaries_groups_metrics():
    results = [
        SimpleNamespace(
            evaluator_type="heuristic-rag-evaluator-v1",
            metric_name="overall_quality_score",
            metric_value=0.9,
        ),
        SimpleNamespace(
            evaluator_type="heuristic-rag-evaluator-v1",
            metric_name="overall_quality_score",
            metric_value=1.0,
        ),
        SimpleNamespace(
            evaluator_type="quality-gate-evaluator-v1",
            metric_name="quality_gate_pass_rate",
            metric_value=1.0,
        ),
    ]

    summaries = build_metric_summaries(results)

    assert len(summaries) == 2

    first = summaries[0]
    assert first.evaluator_type == "heuristic-rag-evaluator-v1"
    assert first.metric_name == "overall_quality_score"
    assert first.count == 2
    assert first.average_value == 0.95
    assert first.min_value == 0.9
    assert first.max_value == 1.0

def test_build_experiment_inspection_summary():
    now = datetime.now(timezone.utc)

    runs = [
        SimpleNamespace(
            workflow_type="rag_answer",
            status="completed",
            latency_ms=100,
            created_at=now,
        ),
        SimpleNamespace(
            workflow_type="tool_calling_agent",
            status="completed",
            latency_ms=200,
            created_at=now,
        ),
        SimpleNamespace(
            workflow_type="tool_calling_agent",
            status="failed",
            latency_ms=None,
            created_at=now,
        ),
    ]

    evaluation_results = [
        SimpleNamespace(
            evaluator_type="agent-evaluator-v1",
            metric_name="overall_agent_score",
            metric_value=1.0,
        )
    ]

    summary = build_experiment_inspection_summary(
        runs=runs,
        evaluation_results=evaluation_results,
    )

    assert summary.total_runs == 3
    assert summary.workflow_type_counts == {
        "rag_answer": 1,
        "tool_calling_agent": 2,
    }
    assert summary.status_counts == {
        "completed": 2,
        "failed": 1,
    }
    assert summary.completed_runs == 2
    assert summary.failed_runs == 1
    assert summary.average_latency_ms == 150.0
    assert summary.latest_run_created_at == now
    assert summary.total_evaluation_results == 1
    assert len(summary.metric_summaries) == 1