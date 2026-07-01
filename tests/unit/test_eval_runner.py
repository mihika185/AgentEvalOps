import json

import pytest

from backend.app.evaluation.eval_runner import (
    EvalRunnerError,
    load_agent_eval_cases,
    summarize_agent_eval_results,
    AgentEvalCaseResult,
)


def test_load_agent_eval_cases_reads_valid_eval_set(tmp_path):
    eval_set_path = tmp_path / "agent_eval_set.json"
    eval_set_path.write_text(
        json.dumps(
            [
                {
                    "id": "agent_case_001",
                    "question": "Calculate 1 + 1",
                    "expected_tools": ["calculator"],
                    "expected_answer_terms": ["2"],
                    "max_allowed_tool_calls": 1,
                    "metadata": {
                        "category": "calculator"
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_agent_eval_cases(str(eval_set_path))

    assert len(cases) == 1
    assert cases[0].case_id == "agent_case_001"
    assert cases[0].question == "Calculate 1 + 1"
    assert cases[0].expected_tools == ["calculator"]
    assert cases[0].expected_answer_terms == ["2"]
    assert cases[0].max_allowed_tool_calls == 1
    assert cases[0].metadata["category"] == "calculator"


def test_load_agent_eval_cases_rejects_missing_question(tmp_path):
    eval_set_path = tmp_path / "agent_eval_set.json"
    eval_set_path.write_text(
        json.dumps(
            [
                {
                    "id": "agent_case_001",
                    "expected_tools": ["calculator"],
                    "expected_answer_terms": ["2"],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvalRunnerError):
        load_agent_eval_cases(str(eval_set_path))


def test_summarize_agent_eval_results_aggregates_metrics():
    summary = summarize_agent_eval_results(
        eval_set_path="data/eval_sets/agent_eval_set.json",
        results=[
            AgentEvalCaseResult(
                case_id="case_1",
                question="Calculate 1 + 1",
                run_id="run_1",
                passed=True,
                failure_reason=None,
                metrics={
                    "tool_selection_accuracy": 1.0,
                    "tool_order_accuracy": 1.0,
                    "answer_correctness": 1.0,
                    "tool_call_success_rate": 1.0,
                    "unnecessary_tool_call_rate": 0.0,
                    "overall_agent_score": 1.0,
                },
                expected_tools=["calculator"],
                actual_tools=["calculator"],
                final_answer="The result is 2.",
                latency_ms=10,
            ),
            AgentEvalCaseResult(
                case_id="case_2",
                question="Check order",
                run_id="run_2",
                passed=False,
                failure_reason="Agent did not select all expected tools",
                metrics={
                    "tool_selection_accuracy": 0.0,
                    "tool_order_accuracy": 0.0,
                    "answer_correctness": 0.0,
                    "tool_call_success_rate": 1.0,
                    "unnecessary_tool_call_rate": 0.0,
                    "overall_agent_score": 0.15,
                },
                expected_tools=["mock_api"],
                actual_tools=["calculator"],
                final_answer="The result is 2.",
                latency_ms=12,
            ),
        ],
    )

    assert summary.total_cases == 2
    assert summary.passed_cases == 1
    assert summary.failed_cases == 1
    assert summary.pass_rate == 0.5
    assert summary.average_tool_selection_accuracy == 0.5
    assert summary.average_tool_order_accuracy == 0.5
    assert summary.average_answer_correctness == 0.5
    assert summary.average_tool_call_success_rate == 1.0
    assert summary.average_unnecessary_tool_call_rate == 0.0
    assert summary.average_overall_agent_score == 0.575