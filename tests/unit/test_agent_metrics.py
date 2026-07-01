from backend.app.evaluation.agent_metrics import (
    AgentToolCallForEvaluation,
    calculate_tool_order_accuracy,
    evaluate_agent_output,
)
from backend.app.evaluation.agent_metrics import extract_terms

def test_extract_terms_splits_hyphenated_terms():
    terms = extract_terms("Customer CUST-1001 has a 12-hour support SLA.")

    assert "cust-1001" in terms
    assert "12-hour" in terms
    assert "12" in terms
    assert "hour" in terms

def test_agent_metrics_pass_for_correct_calculator_call():
    summary = evaluate_agent_output(
        expected_tools=["calculator"],
        expected_answer_terms=["18.4"],
        actual_tool_calls=[
            AgentToolCallForEvaluation(
                tool_name="calculator",
                input_data={"expression": "(142080 - 120000) / 120000 * 100"},
                output_data={"result": 18.4},
                success=True,
            )
        ],
        final_answer="The calculated result is 18.4.",
        max_allowed_tool_calls=1,
    )

    metrics = {
        metric.metric_name: metric.metric_value
        for metric in summary.metrics
    }

    assert summary.passed is True
    assert metrics["tool_selection_accuracy"] == 1.0
    assert metrics["tool_order_accuracy"] == 1.0
    assert metrics["answer_correctness"] == 1.0
    assert metrics["tool_call_success_rate"] == 1.0
    assert metrics["overall_agent_score"] == 1.0

def test_agent_metrics_detect_missing_expected_tool():
    summary = evaluate_agent_output(
        expected_tools=["document_search"],
        expected_answer_terms=["chunk"],
        actual_tool_calls=[
            AgentToolCallForEvaluation(
                tool_name="calculator",
                input_data={"expression": "1 + 1"},
                output_data={"result": 2},
                success=True,
            )
        ],
        final_answer="The calculated result is 2.",
        max_allowed_tool_calls=1,
    )

    metrics = {
        metric.metric_name: metric.metric_value
        for metric in summary.metrics
    }

    assert summary.passed is False
    assert metrics["tool_selection_accuracy"] == 0.0
    assert "Agent did not select all expected tools" in summary.failure_reasons

def test_agent_metrics_detect_extra_tool_calls():
    summary = evaluate_agent_output(
        expected_tools=["mock_api"],
        expected_answer_terms=["delivered"],
        actual_tool_calls=[
            AgentToolCallForEvaluation(
                tool_name="mock_api",
                input_data={"endpoint": "get_order_status"},
                output_data={"status": "delivered"},
                success=True,
            ),
            AgentToolCallForEvaluation(
                tool_name="document_search",
                input_data={"query": "extra search"},
                output_data={},
                success=True,
            ),
        ],
        final_answer="Order ORD-1001 is currently delivered.",
        max_allowed_tool_calls=1,
    )

    metrics = {
        metric.metric_name: metric.metric_value
        for metric in summary.metrics
    }

    assert summary.passed is False
    assert metrics["unnecessary_tool_call_rate"] == 0.5
    assert metrics["max_tool_call_score"] == 0.0
    assert "Agent exceeded the maximum allowed tool calls" in summary.failure_reasons

def test_tool_order_accuracy_uses_lcs_ratio():
    score = calculate_tool_order_accuracy(
        expected_tools=["sql_query", "calculator"],
        actual_tools=["calculator", "sql_query"],
    )

    assert score == 0.5

def test_agent_metrics_detect_failed_tool_call():
    summary = evaluate_agent_output(
        expected_tools=["mock_api"],
        expected_answer_terms=["delivered"],
        actual_tool_calls=[
            AgentToolCallForEvaluation(
                tool_name="mock_api",
                input_data={"endpoint": "get_order_status"},
                output_data={},
                success=False,
                error_message="No mock order found",
            )
        ],
        final_answer="No mock order found",
        max_allowed_tool_calls=1,
    )

    metrics = {
        metric.metric_name: metric.metric_value
        for metric in summary.metrics
    }

    assert summary.passed is False
    assert metrics["tool_call_success_rate"] == 0.0
    assert "One or more tool calls failed" in summary.failure_reasons