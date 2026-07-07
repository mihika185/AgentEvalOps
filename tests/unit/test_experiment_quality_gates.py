from backend.app.evaluation.experiment_quality_gates import (
    EXPERIMENT_QUALITY_GATE_PROFILES,
    evaluate_aggregate_quality_gates,
)


def valid_experiment_metrics():
    return {
        "total_runs": 5.0,
        "completed_runs": 5.0,
        "failed_runs": 0.0,
        "completed_run_rate": 1.0,
        "failed_run_rate": 0.0,
        "average_overall_quality_score": 0.91,
        "average_hallucination_risk": 0.04,
        "average_latency_ms": 1200.0,
        "average_estimated_cost": 0.01,
    }

def test_experiment_quality_gates_pass_for_good_metrics():
    summary = evaluate_aggregate_quality_gates(
        scope_type="experiment",
        scope_id="exp_test",
        profile_name="default-v1",
        aggregate_metrics=valid_experiment_metrics(),
        gates=EXPERIMENT_QUALITY_GATE_PROFILES["default-v1"],
    )

    assert summary.overall_passed is True
    assert summary.failed_count == 0
    assert summary.pass_rate == 1.0

def test_experiment_quality_gates_fail_when_threshold_is_not_met():
    metrics = valid_experiment_metrics()
    metrics["average_hallucination_risk"] = 0.40

    summary = evaluate_aggregate_quality_gates(
        scope_type="experiment",
        scope_id="exp_test",
        profile_name="default-v1",
        aggregate_metrics=metrics,
        gates=EXPERIMENT_QUALITY_GATE_PROFILES["default-v1"],
    )

    failed_checks = [
        check
        for check in summary.checks
        if not check.passed
    ]

    assert summary.overall_passed is False
    assert failed_checks[0].gate_name == "Maximum Average Hallucination Risk"
    assert failed_checks[0].failure_reason == "threshold_not_met"

def test_experiment_quality_gates_fail_when_required_metric_is_missing():
    metrics = valid_experiment_metrics()
    del metrics["average_latency_ms"]

    summary = evaluate_aggregate_quality_gates(
        scope_type="experiment",
        scope_id="exp_test",
        profile_name="default-v1",
        aggregate_metrics=metrics,
        gates=EXPERIMENT_QUALITY_GATE_PROFILES["default-v1"],
    )

    failed_checks = [
        check
        for check in summary.checks
        if not check.passed
    ]

    assert summary.overall_passed is False
    assert failed_checks[0].gate_name == "Maximum Average Latency"
    assert failed_checks[0].failure_reason == "metric_missing"