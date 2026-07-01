import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from sqlalchemy.orm import Session

from backend.app.agents.agent_runner import AgentRunnerError, run_tool_calling_agent
from backend.app.evaluation.agent_metrics import (
    AgentEvaluationError,
    AgentEvaluationSummary,
    evaluate_agent_run,
)

class EvalRunnerError(Exception):
    pass

@dataclass(frozen=True)
class AgentEvalCase:
    case_id: str
    question: str
    expected_tools: list[str]
    expected_answer_terms: list[str]
    max_allowed_tool_calls: Optional[int]
    metadata: dict[str, Any]

@dataclass(frozen=True)
class AgentEvalCaseResult:
    case_id: str
    question: str
    run_id: Optional[str]
    passed: bool
    failure_reason: Optional[str]
    metrics: dict[str, float]
    expected_tools: list[str]
    actual_tools: list[str]
    final_answer: Optional[str]
    latency_ms: Optional[int]

@dataclass(frozen=True)
class AgentEvalRunSummary:
    eval_set_path: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    average_tool_selection_accuracy: float
    average_tool_order_accuracy: float
    average_answer_correctness: float
    average_tool_call_success_rate: float
    average_unnecessary_tool_call_rate: float
    average_overall_agent_score: float
    results: list[AgentEvalCaseResult]

def load_agent_eval_cases(eval_set_path: str) -> list[AgentEvalCase]:
    path = Path(eval_set_path)

    if not path.exists():
        raise EvalRunnerError(f"Agent eval set was not found: {eval_set_path}")

    try:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalRunnerError(f"Agent eval set is not valid JSON: {exc}") from exc

    if not isinstance(raw_cases, list):
        raise EvalRunnerError("Agent eval set must be a JSON list")

    cases = []

    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise EvalRunnerError(f"Agent eval case #{index} must be an object")

        case_id = str(raw_case.get("id", "")).strip()
        question = str(raw_case.get("question", "")).strip()
        expected_tools = raw_case.get("expected_tools", [])
        expected_answer_terms = raw_case.get("expected_answer_terms", [])
        max_allowed_tool_calls = raw_case.get("max_allowed_tool_calls")
        metadata = raw_case.get("metadata", {})

        if not case_id:
            raise EvalRunnerError(f"Agent eval case #{index} is missing id")

        if not question:
            raise EvalRunnerError(f"Agent eval case '{case_id}' is missing question")

        if not isinstance(expected_tools, list):
            raise EvalRunnerError(
                f"Agent eval case '{case_id}' expected_tools must be a list"
            )

        if not isinstance(expected_answer_terms, list):
            raise EvalRunnerError(
                f"Agent eval case '{case_id}' expected_answer_terms must be a list"
            )

        if max_allowed_tool_calls is not None:
            max_allowed_tool_calls = int(max_allowed_tool_calls)

            if max_allowed_tool_calls <= 0:
                raise EvalRunnerError(
                    f"Agent eval case '{case_id}' max_allowed_tool_calls must be positive"
                )

        if not isinstance(metadata, dict):
            raise EvalRunnerError(
                f"Agent eval case '{case_id}' metadata must be an object"
            )

        cases.append(
            AgentEvalCase(
                case_id=case_id,
                question=question,
                expected_tools=[
                    str(tool).strip()
                    for tool in expected_tools
                    if str(tool).strip()
                ],
                expected_answer_terms=[
                    str(term).strip()
                    for term in expected_answer_terms
                    if str(term).strip()
                ],
                max_allowed_tool_calls=max_allowed_tool_calls,
                metadata=metadata,
            )
        )

    return cases

def run_agent_eval_set(
    db: Session,
    eval_set_path: str = "data/eval_sets/agent_eval_set.json",
    document_id: Optional[str] = None,
    retrieval_provider: str = "hybrid",
    top_k: int = 3,
    rerank: bool = True,
    candidate_multiplier: int = 3,
    max_steps: int = 5,
) -> AgentEvalRunSummary:
    cases = load_agent_eval_cases(eval_set_path)
    results: list[AgentEvalCaseResult] = []

    for case in cases:
        try:
            result = run_tool_calling_agent(
                db=db,
                query=case.question,
                document_id=document_id,
                retrieval_provider=retrieval_provider,
                top_k=top_k,
                rerank=rerank,
                candidate_multiplier=candidate_multiplier,
                max_steps=max_steps,
            )

            evaluation = evaluate_agent_run(
                db=db,
                run_id=result.run_id,
                expected_tools=case.expected_tools,
                expected_answer_terms=case.expected_answer_terms,
                max_allowed_tool_calls=case.max_allowed_tool_calls,
                persist=True,
            )

            metrics = metrics_to_dict(evaluation)

            results.append(
                AgentEvalCaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    run_id=result.run_id,
                    passed=evaluation.passed,
                    failure_reason=build_failure_reason(evaluation),
                    metrics=metrics,
                    expected_tools=case.expected_tools,
                    actual_tools=[
                        tool_call.tool_name
                        for tool_call in result.tool_calls
                    ],
                    final_answer=result.final_answer,
                    latency_ms=result.total_latency_ms,
                )
            )

        except (AgentRunnerError, AgentEvaluationError, EvalRunnerError) as exc:
            results.append(
                AgentEvalCaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    run_id=None,
                    passed=False,
                    failure_reason=str(exc),
                    metrics={},
                    expected_tools=case.expected_tools,
                    actual_tools=[],
                    final_answer=None,
                    latency_ms=None,
                )
            )

    return summarize_agent_eval_results(
        eval_set_path=eval_set_path,
        results=results,
    )

def summarize_agent_eval_results(
    eval_set_path: str,
    results: list[AgentEvalCaseResult],
) -> AgentEvalRunSummary:
    total_cases = len(results)
    passed_cases = sum(1 for result in results if result.passed)
    failed_cases = total_cases - passed_cases

    return AgentEvalRunSummary(
        eval_set_path=eval_set_path,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        pass_rate=round(safe_ratio(passed_cases, total_cases), 4),
        average_tool_selection_accuracy=average_metric(
            results,
            "tool_selection_accuracy",
        ),
        average_tool_order_accuracy=average_metric(
            results,
            "tool_order_accuracy",
        ),
        average_answer_correctness=average_metric(
            results,
            "answer_correctness",
        ),
        average_tool_call_success_rate=average_metric(
            results,
            "tool_call_success_rate",
        ),
        average_unnecessary_tool_call_rate=average_metric(
            results,
            "unnecessary_tool_call_rate",
        ),
        average_overall_agent_score=average_metric(
            results,
            "overall_agent_score",
        ),
        results=results,
    )

def metrics_to_dict(summary: AgentEvaluationSummary) -> dict[str, float]:
    return {
        metric.metric_name: metric.metric_value
        for metric in summary.metrics
    }

def build_failure_reason(summary: AgentEvaluationSummary) -> Optional[str]:
    if summary.passed:
        return None

    if not summary.failure_reasons:
        return "Agent evaluation failed"

    return "; ".join(summary.failure_reasons)

def average_metric(
    results: list[AgentEvalCaseResult],
    metric_name: str,
) -> float:
    values = [
        result.metrics[metric_name]
        for result in results
        if metric_name in result.metrics
    ]

    if not values:
        return 0.0

    return round(sum(values) / len(values), 4)

def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator