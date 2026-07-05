from types import SimpleNamespace

from backend.app.api.runs import build_run_inspection_summary


def test_build_run_inspection_summary_counts_trace_and_evaluations():
    trace_steps = [
        SimpleNamespace(status="completed", latency_ms=10),
        SimpleNamespace(status="failed", latency_ms=5),
        SimpleNamespace(status="completed", latency_ms=None),
    ]

    evaluation_results = [
        SimpleNamespace(evaluator_type="rag-answer-evaluator-v1"),
        SimpleNamespace(evaluator_type="agent-evaluator-v1"),
    ]

    quality_gate_results = [
        SimpleNamespace(evaluator_type="quality-gate-evaluator-v1"),
    ]

    summary = build_run_inspection_summary(
        trace_steps=trace_steps,
        evaluation_results=evaluation_results,
        quality_gate_results=quality_gate_results,
    )

    assert summary.trace_step_count == 3
    assert summary.evaluation_result_count == 2
    assert summary.quality_gate_result_count == 1
    assert summary.failed_trace_step_count == 1
    assert summary.total_trace_latency_ms == 15
    assert summary.evaluator_types == [
        "agent-evaluator-v1",
        "quality-gate-evaluator-v1",
        "rag-answer-evaluator-v1",
    ]