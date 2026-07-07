from datetime import datetime, timezone
from types import SimpleNamespace

from backend.app.reporting.aggregate_report_service import (
    build_readiness_decision,
    categorize_benchmark_failure,
    summarize_evaluation_results,
    summarize_runs,
)


def test_summarize_runs_includes_latency_tokens_and_cost():
    runs = [
        SimpleNamespace(
            status="completed",
            latency_ms=100,
            prompt_tokens=20,
            completion_tokens=5,
            estimated_cost=0.01,
            metadata_json={"total_tokens": 25},
        ),
        SimpleNamespace(
            status="failed",
            latency_ms=300,
            prompt_tokens=40,
            completion_tokens=10,
            estimated_cost=0.03,
            metadata_json={"total_tokens": 50},
        ),
    ]

    summary = summarize_runs(runs)

    assert summary["total_runs"] == 2.0
    assert summary["completed_runs"] == 1.0
    assert summary["failed_runs"] == 1.0
    assert summary["completed_run_rate"] == 0.5
    assert summary["failed_run_rate"] == 0.5
    assert summary["average_latency_ms"] == 200.0
    assert summary["average_prompt_tokens"] == 30.0
    assert summary["average_completion_tokens"] == 7.5
    assert summary["average_total_tokens"] == 37.5
    assert summary["average_estimated_cost"] == 0.02
    assert summary["total_estimated_cost"] == 0.04


def test_summarize_evaluation_results_groups_metric_values():
    evaluation_results = [
        SimpleNamespace(
            evaluator_type="heuristic-rag-evaluator-v2",
            metric_name="overall_quality_score",
            metric_value=0.8,
        ),
        SimpleNamespace(
            evaluator_type="heuristic-rag-evaluator-v2",
            metric_name="overall_quality_score",
            metric_value=1.0,
        ),
    ]

    summaries = summarize_evaluation_results(evaluation_results)

    assert summaries == [
        {
            "evaluator_type": "heuristic-rag-evaluator-v2",
            "metric_name": "overall_quality_score",
            "count": 2,
            "average_value": 0.9,
            "min_value": 0.8,
            "max_value": 1.0,
        }
    ]


def test_readiness_decision_is_ready_when_no_risks_exist():
    decision = build_readiness_decision(
        scope_type="experiment",
        gate_overall_passed=True,
        failed_count=0,
        failure_rate=0.0,
        average_hallucination_risk=0.0,
    )

    assert decision["ready"] is True
    assert decision["status"] == "ready"
    assert decision["reasons"] == ["all_release_checks_passed"]


def test_readiness_decision_blocks_when_gate_fails():
    decision = build_readiness_decision(
        scope_type="benchmark_run",
        gate_overall_passed=False,
        failed_count=2,
        failure_rate=0.25,
        average_hallucination_risk=0.30,
    )

    assert decision["ready"] is False
    assert decision["status"] == "needs_attention"
    assert "aggregate_quality_gates_failed" in decision["reasons"]
    assert "failures_present" in decision["reasons"]
    assert "hallucination_risk_above_release_bar" in decision["reasons"]


def test_categorize_benchmark_failure_detects_missing_keywords():
    item = SimpleNamespace(
        passed=False,
        failure_reason="Answer is missing expected keywords: refund",
        metadata_json={},
    )

    assert categorize_benchmark_failure(item) == "missing_expected_keywords"