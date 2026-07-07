from types import SimpleNamespace

from backend.app.reporting.dashboard_service import (
    build_latency_cost_summary,
    build_quality_summary,
    build_run_health,
    benchmark_run_to_dashboard_item,
)


def make_run(
    status: str,
    latency_ms: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    estimated_cost: float | None = None,
):
    return SimpleNamespace(
        id="run_test",
        experiment_id="exp_test",
        workflow_type="rag_answer",
        status=status,
        input_query="Question",
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost=estimated_cost,
        metadata_json={},
        created_at=None,
        completed_at=None,
    )


def test_build_run_health_counts_statuses():
    summary = build_run_health(
        [
            make_run("completed"),
            make_run("completed"),
            make_run("failed"),
        ]
    )

    assert summary["total_runs"] == 3
    assert summary["completed_runs"] == 2
    assert summary["failed_runs"] == 1
    assert summary["completed_run_rate"] == 0.6667
    assert summary["failed_run_rate"] == 0.3333


def test_build_latency_cost_summary_aggregates_values():
    summary = build_latency_cost_summary(
        [
            make_run(
                status="completed",
                latency_ms=100,
                prompt_tokens=20,
                completion_tokens=5,
                estimated_cost=0.01,
            ),
            make_run(
                status="completed",
                latency_ms=300,
                prompt_tokens=40,
                completion_tokens=15,
                estimated_cost=0.03,
            ),
        ]
    )

    assert summary["average_latency_ms"] == 200.0
    assert summary["max_latency_ms"] == 300.0
    assert summary["average_prompt_tokens"] == 30.0
    assert summary["average_completion_tokens"] == 10.0
    assert summary["average_total_tokens"] == 40.0
    assert summary["average_estimated_cost"] == 0.02
    assert summary["total_estimated_cost"] == 0.04


def test_build_quality_summary_groups_metrics():
    evaluation_results = [
        SimpleNamespace(
            metric_name="overall_quality_score",
            metric_value=0.8,
        ),
        SimpleNamespace(
            metric_name="overall_quality_score",
            metric_value=1.0,
        ),
        SimpleNamespace(
            metric_name="hallucination_risk",
            metric_value=0.2,
        ),
    ]

    summary = build_quality_summary(evaluation_results)

    assert summary["average_overall_quality_score"] == 0.9
    assert summary["average_hallucination_risk"] == 0.2
    assert summary["metric_result_count"] == 3


def test_benchmark_run_to_dashboard_item_includes_pass_rate():
    benchmark_run = SimpleNamespace(
        id="benchrun_test",
        dataset_id="dataset_test",
        status="completed",
        total_cases=10,
        passed_cases=8,
        failed_cases=2,
        average_overall_quality_score=0.85,
        average_hallucination_risk=0.1,
        average_latency_ms=450.0,
        metadata_json={
            "pipeline_config_id": "pipe_test",
            "pipeline_config_name": "Hybrid Pipeline",
        },
        started_at=None,
        completed_at=None,
    )

    item = benchmark_run_to_dashboard_item(benchmark_run)

    assert item["pass_rate"] == 0.8
    assert item["pipeline_config_id"] == "pipe_test"
    assert item["pipeline_config_name"] == "Hybrid Pipeline"