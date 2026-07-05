from types import SimpleNamespace

from scripts.generate_benchmark_report import (
    average_latency,
    count_by_attribute,
    format_value,
    metric_by_prefix,
    pass_rate,
    pipeline_name,
    summarize_metric_values,
    truncate,
)

def test_pass_rate_calculates_completed_case_ratio():
    run = SimpleNamespace(
        passed_cases=3,
        total_cases=4,
    )

    assert pass_rate(run) == 0.75

def test_pipeline_name_prefers_pipeline_config_name():
    run = SimpleNamespace(
        metadata_json={
            "pipeline_config_name": "Hybrid Cross-Encoder Rerank top-k-3"
        }
    )

    assert pipeline_name(run) == "Hybrid Cross-Encoder Rerank top-k-3"

def test_count_by_attribute_counts_values():
    runs = [
        SimpleNamespace(workflow_type="rag_answer"),
        SimpleNamespace(workflow_type="tool_calling_agent"),
        SimpleNamespace(workflow_type="rag_answer"),
    ]

    assert count_by_attribute(runs, "workflow_type") == {
        "rag_answer": 2,
        "tool_calling_agent": 1,
    }

def test_summarize_metric_values_groups_by_evaluator_and_metric():
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

    summaries = summarize_metric_values(results)

    assert summaries[0] == {
        "evaluator_type": "heuristic-rag-evaluator-v1",
        "metric_name": "overall_quality_score",
        "count": 2,
        "average_value": 0.95,
        "min_value": 0.9,
        "max_value": 1.0,
    }

def test_metric_by_prefix_returns_first_matching_metric():
    metrics = {
        "overall_quality_score": 0.95,
        "recall_at_3": 1.0,
    }

    assert metric_by_prefix(metrics, "recall_at_") == 1.0

def test_average_latency_ignores_none_values():
    assert average_latency([100, None, 200]) == 150.0


def test_format_value_handles_none_and_floats():
    assert format_value(None) == "-"
    assert format_value(0.333333) == "0.3333"
    assert format_value(1.0) == "1"


def test_truncate_shortens_long_text():
    text = "What must customers provide when reporting a damaged product?"

    assert truncate(text, 20) == "What must custome..."
    assert len(truncate(text, 20)) == 20