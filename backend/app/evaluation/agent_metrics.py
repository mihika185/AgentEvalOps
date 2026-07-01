import re
from dataclasses import dataclass
from typing import Any, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import EvaluationResult, Run, TraceStep
from backend.app.logging_config import get_logger


logger = get_logger(__name__)

AGENT_EVALUATOR_TYPE = "heuristic-agent-evaluator-v1"

class AgentEvaluationError(Exception):
    pass

@dataclass(frozen=True)
class AgentMetricResult:
    metric_name: str
    metric_value: float
    details: dict[str, Any]

@dataclass(frozen=True)
class AgentEvaluationSummary:
    run_id: str
    evaluator_type: str
    metrics: list[AgentMetricResult]
    passed: bool
    failure_reasons: list[str]

@dataclass(frozen=True)
class AgentToolCallForEvaluation:
    tool_name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    success: bool
    error_message: Optional[str] = None

def evaluate_agent_output(
    expected_tools: list[str],
    expected_answer_terms: list[str],
    actual_tool_calls: list[AgentToolCallForEvaluation],
    final_answer: str,
    max_allowed_tool_calls: Optional[int] = None,
) -> AgentEvaluationSummary:
    actual_tools = [
        tool_call.tool_name
        for tool_call in actual_tool_calls
    ]

    cleaned_expected_tools = normalize_tool_names(expected_tools)
    cleaned_expected_terms = normalize_terms(expected_answer_terms)
    answer_terms = extract_terms(final_answer)

    tool_selection_accuracy = calculate_tool_selection_accuracy(
        expected_tools=cleaned_expected_tools,
        actual_tools=actual_tools,
    )

    tool_order_accuracy = calculate_tool_order_accuracy(
        expected_tools=cleaned_expected_tools,
        actual_tools=actual_tools,
    )

    unnecessary_tool_call_rate = calculate_unnecessary_tool_call_rate(
        expected_tools=cleaned_expected_tools,
        actual_tools=actual_tools,
    )

    tool_efficiency_score = 1.0 - unnecessary_tool_call_rate

    tool_call_success_rate = calculate_tool_call_success_rate(
        expected_tools=cleaned_expected_tools,
        actual_tool_calls=actual_tool_calls,
    )

    answer_correctness = calculate_answer_correctness(
        expected_terms=cleaned_expected_terms,
        answer_terms=answer_terms,
    )

    max_tool_call_score = calculate_max_tool_call_score(
        actual_tool_count=len(actual_tool_calls),
        max_allowed_tool_calls=max_allowed_tool_calls,
    )

    overall_agent_score = (
        0.30 * tool_selection_accuracy
        + 0.20 * tool_order_accuracy
        + 0.25 * answer_correctness
        + 0.15 * tool_call_success_rate
        + 0.10 * tool_efficiency_score
    )

    overall_agent_score = min(overall_agent_score, max_tool_call_score)

    metrics = [
        AgentMetricResult(
            metric_name="tool_selection_accuracy",
            metric_value=round(tool_selection_accuracy, 4),
            details={
                "expected_tools": cleaned_expected_tools,
                "actual_tools": actual_tools,
            },
        ),
        AgentMetricResult(
            metric_name="tool_order_accuracy",
            metric_value=round(tool_order_accuracy, 4),
            details={
                "expected_tools": cleaned_expected_tools,
                "actual_tools": actual_tools,
                "definition": "Longest common subsequence ratio against expected tool order",
            },
        ),
        AgentMetricResult(
            metric_name="answer_correctness",
            metric_value=round(answer_correctness, 4),
            details={
                "expected_answer_terms": cleaned_expected_terms,
                "matched_answer_terms": sorted(
                    set(cleaned_expected_terms).intersection(answer_terms)
                ),
                "answer_terms": sorted(answer_terms),
            },
        ),
        AgentMetricResult(
            metric_name="tool_call_success_rate",
            metric_value=round(tool_call_success_rate, 4),
            details={
                "total_tool_calls": len(actual_tool_calls),
                "successful_tool_calls": sum(
                    1 for tool_call in actual_tool_calls if tool_call.success
                ),
            },
        ),
        AgentMetricResult(
            metric_name="unnecessary_tool_call_rate",
            metric_value=round(unnecessary_tool_call_rate, 4),
            details={
                "expected_tool_count": len(cleaned_expected_tools),
                "actual_tool_count": len(actual_tools),
            },
        ),
        AgentMetricResult(
            metric_name="tool_efficiency_score",
            metric_value=round(tool_efficiency_score, 4),
            details={
                "definition": "1 - unnecessary_tool_call_rate",
            },
        ),
        AgentMetricResult(
            metric_name="max_tool_call_score",
            metric_value=round(max_tool_call_score, 4),
            details={
                "max_allowed_tool_calls": max_allowed_tool_calls,
                "actual_tool_count": len(actual_tool_calls),
            },
        ),
        AgentMetricResult(
            metric_name="overall_agent_score",
            metric_value=round(overall_agent_score, 4),
            details={
                "weights": {
                    "tool_selection_accuracy": 0.30,
                    "tool_order_accuracy": 0.20,
                    "answer_correctness": 0.25,
                    "tool_call_success_rate": 0.15,
                    "tool_efficiency_score": 0.10,
                },
                "max_tool_call_score_applied": max_tool_call_score,
            },
        ),
    ]

    failure_reasons = build_failure_reasons(
        tool_selection_accuracy=tool_selection_accuracy,
        tool_order_accuracy=tool_order_accuracy,
        answer_correctness=answer_correctness,
        tool_call_success_rate=tool_call_success_rate,
        max_tool_call_score=max_tool_call_score,
    )

    return AgentEvaluationSummary(
        run_id="in_memory",
        evaluator_type=AGENT_EVALUATOR_TYPE,
        metrics=metrics,
        passed=not failure_reasons,
        failure_reasons=failure_reasons,
    )

def evaluate_agent_run(
    db: Session,
    run_id: str,
    expected_tools: list[str],
    expected_answer_terms: list[str],
    max_allowed_tool_calls: Optional[int] = None,
    persist: bool = True,
) -> AgentEvaluationSummary:
    run = db.get(Run, run_id)

    if run is None:
        raise AgentEvaluationError(f"Run with id '{run_id}' was not found")

    if run.workflow_type != "tool_calling_agent":
        raise AgentEvaluationError(
            f"Run '{run_id}' has unsupported workflow_type '{run.workflow_type}'"
        )

    if run.status != "completed":
        raise AgentEvaluationError(
            f"Run '{run_id}' must be completed before evaluation"
        )

    tool_calls = get_agent_tool_calls(db=db, run_id=run_id)
    final_answer = get_agent_final_answer(db=db, run_id=run_id, run=run)

    summary = evaluate_agent_output(
        expected_tools=expected_tools,
        expected_answer_terms=expected_answer_terms,
        actual_tool_calls=tool_calls,
        final_answer=final_answer,
        max_allowed_tool_calls=max_allowed_tool_calls,
    )

    summary = AgentEvaluationSummary(
        run_id=run_id,
        evaluator_type=summary.evaluator_type,
        metrics=summary.metrics,
        passed=summary.passed,
        failure_reasons=summary.failure_reasons,
    )

    if persist:
        save_agent_metrics(db=db, run_id=run_id, summary=summary)

    logger.info(
        "Evaluated agent run %s with %s metrics",
        run_id,
        len(summary.metrics),
    )

    return summary

def get_agent_tool_calls(
    db: Session,
    run_id: str,
) -> list[AgentToolCallForEvaluation]:
    trace_steps = db.execute(
        select(TraceStep)
        .where(
            TraceStep.run_id == run_id,
            TraceStep.step_type == "tool_call",
        )
        .order_by(TraceStep.step_index.asc())
    ).scalars().all()

    return [
        AgentToolCallForEvaluation(
            tool_name=step.name,
            input_data=step.input_data or {},
            output_data=step.output_data or {},
            success=step.status == "completed",
            error_message=step.error_message,
        )
        for step in trace_steps
    ]


def get_agent_final_answer(
    db: Session,
    run_id: str,
    run: Run,
) -> str:
    final_step = db.execute(
        select(TraceStep)
        .where(
            TraceStep.run_id == run_id,
            TraceStep.step_type == "agent_final_response",
        )
        .order_by(TraceStep.step_index.desc())
    ).scalars().first()

    if final_step is not None:
        final_answer = final_step.output_data.get("final_answer")

        if final_answer:
            return str(final_answer)

    return run.output_answer or ""

def save_agent_metrics(
    db: Session,
    run_id: str,
    summary: AgentEvaluationSummary,
) -> None:
    existing_results = db.execute(
        select(EvaluationResult)
        .where(
            EvaluationResult.run_id == run_id,
            EvaluationResult.evaluator_type == AGENT_EVALUATOR_TYPE,
        )
    ).scalars().all()

    for result in existing_results:
        db.delete(result)

    for metric in summary.metrics:
        db.add(
            EvaluationResult(
                run_id=run_id,
                metric_name=metric.metric_name,
                metric_value=metric.metric_value,
                evaluator_type=AGENT_EVALUATOR_TYPE,
                details_json=metric.details,
            )
        )

    db.add(
        EvaluationResult(
            run_id=run_id,
            metric_name="agent_eval_passed",
            metric_value=1.0 if summary.passed else 0.0,
            evaluator_type=AGENT_EVALUATOR_TYPE,
            details_json={
                "failure_reasons": summary.failure_reasons,
            },
        )
    )

    run = db.get(Run, run_id)

    if run is not None:
        run.metadata_json = {
            **(run.metadata_json or {}),
            "agent_eval_passed": summary.passed,
            "agent_eval_failure_reasons": summary.failure_reasons,
        }

    db.commit()

def calculate_tool_selection_accuracy(
    expected_tools: list[str],
    actual_tools: list[str],
) -> float:
    if not expected_tools:
        return 1.0 if not actual_tools else 0.0

    matched_count = 0
    remaining_actual_tools = list(actual_tools)

    for expected_tool in expected_tools:
        if expected_tool in remaining_actual_tools:
            matched_count += 1
            remaining_actual_tools.remove(expected_tool)

    return matched_count / len(expected_tools)

def calculate_tool_order_accuracy(
    expected_tools: list[str],
    actual_tools: list[str],
) -> float:
    if not expected_tools:
        return 1.0 if not actual_tools else 0.0

    lcs_length = longest_common_subsequence_length(
        expected_tools,
        actual_tools,
    )

    return lcs_length / len(expected_tools)

def calculate_unnecessary_tool_call_rate(
    expected_tools: list[str],
    actual_tools: list[str],
) -> float:
    if not actual_tools:
        return 0.0

    extra_tool_count = max(0, len(actual_tools) - len(expected_tools))

    return extra_tool_count / len(actual_tools)

def calculate_tool_call_success_rate(
    expected_tools: list[str],
    actual_tool_calls: list[AgentToolCallForEvaluation],
) -> float:
    if not actual_tool_calls:
        return 1.0 if not expected_tools else 0.0

    successful_tool_calls = [
        tool_call
        for tool_call in actual_tool_calls
        if tool_call.success
    ]

    return len(successful_tool_calls) / len(actual_tool_calls)

def calculate_answer_correctness(
    expected_terms: list[str],
    answer_terms: set[str],
) -> float:
    if not expected_terms:
        return 1.0

    matched_terms = set(expected_terms).intersection(answer_terms)

    return len(matched_terms) / len(set(expected_terms))


def calculate_max_tool_call_score(
    actual_tool_count: int,
    max_allowed_tool_calls: Optional[int],
) -> float:
    if max_allowed_tool_calls is None:
        return 1.0

    if actual_tool_count <= max_allowed_tool_calls:
        return 1.0

    return 0.0

def longest_common_subsequence_length(
    expected_tools: list[str],
    actual_tools: list[str],
) -> int:
    rows = len(expected_tools) + 1
    cols = len(actual_tools) + 1

    table = [
        [0 for _ in range(cols)]
        for _ in range(rows)
    ]

    for row in range(1, rows):
        for col in range(1, cols):
            if expected_tools[row - 1] == actual_tools[col - 1]:
                table[row][col] = table[row - 1][col - 1] + 1
            else:
                table[row][col] = max(
                    table[row - 1][col],
                    table[row][col - 1],
                )

    return table[-1][-1]

def build_failure_reasons(
    tool_selection_accuracy: float,
    tool_order_accuracy: float,
    answer_correctness: float,
    tool_call_success_rate: float,
    max_tool_call_score: float,
) -> list[str]:
    reasons = []

    if tool_selection_accuracy < 1.0:
        reasons.append("Agent did not select all expected tools")
    if tool_order_accuracy < 1.0:
        reasons.append("Agent did not call tools in the expected order")
    if answer_correctness < 1.0:
        reasons.append("Final answer is missing expected terms")
    if tool_call_success_rate < 1.0:
        reasons.append("One or more tool calls failed")
    if max_tool_call_score < 1.0:
        reasons.append("Agent exceeded the maximum allowed tool calls")

    return reasons

def normalize_tool_names(tool_names: list[str]) -> list[str]:
    return [
        tool_name.strip().lower()
        for tool_name in tool_names
        if tool_name.strip()
    ]

def normalize_terms(terms: list[str]) -> list[str]:
    normalized_terms = []

    for term in terms:
        cleaned_term = normalize_text(term)

        if cleaned_term:
            normalized_terms.append(cleaned_term)

    return normalized_terms

def extract_terms(text: str) -> set[str]:
    raw_terms = re.findall(r"[a-zA-Z0-9.\-]+", text.lower())

    terms = set()

    for raw_term in raw_terms:
        normalized_term = normalize_text(raw_term)

        if not normalized_term:
            continue

        terms.add(normalized_term)

        if "-" in normalized_term:
            for part in normalized_term.split("-"):
                cleaned_part = normalize_text(part)

                if cleaned_part:
                    terms.add(cleaned_part)

    return terms

def normalize_text(text: str) -> str:
    return text.strip().lower().strip(".,;:!?()[]{}\"'")