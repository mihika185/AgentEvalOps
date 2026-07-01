import re
import time
from dataclasses import dataclass
from typing import Any, Optional
from sqlalchemy.orm import Session

from backend.app.agents.tool_registry import build_tool_registry
from backend.app.agents.tools.base_tool import ToolResult
from backend.app.logging_config import get_logger
from backend.app.observability.run_recorder import (
    RunRecorderError,
    complete_run,
    create_run,
    fail_run,
    record_trace_step,
)

logger = get_logger(__name__)

class AgentRunnerError(Exception):
    pass

@dataclass(frozen=True)
class AgentToolCall:
    tool_name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    success: bool
    error_message: Optional[str] = None

@dataclass(frozen=True)
class AgentRunResult:
    run_id: str
    query: str
    final_answer: str
    status: str
    tool_calls: list[AgentToolCall]
    total_latency_ms: int
    metadata: dict[str, Any]

def run_tool_calling_agent(
    db: Session,
    query: str,
    document_id: Optional[str] = None,
    retrieval_provider: str = "hybrid",
    top_k: int = 3,
    rerank: bool = True,
    candidate_multiplier: int = 3,
    max_steps: int = 5,
) -> AgentRunResult:
    cleaned_query = query.strip()

    if not cleaned_query:
        raise AgentRunnerError("query cannot be empty")

    if max_steps <= 0:
        raise AgentRunnerError("max_steps must be greater than 0")

    total_start = time.perf_counter()
    run_id: Optional[str] = None

    try:
        run = create_run(
            db=db,
            workflow_type="tool_calling_agent",
            input_query=cleaned_query,
            metadata={
                "agent_version": "rule-based-agent-v1",
                "document_id": document_id,
                "retrieval_provider": retrieval_provider,
                "top_k": top_k,
                "rerank": rerank,
                "candidate_multiplier": candidate_multiplier,
                "max_steps": max_steps,
            },
        )

        run_id = run.run_id

        plan = build_agent_plan(
            query=cleaned_query,
            document_id=document_id,
            retrieval_provider=retrieval_provider,
            top_k=top_k,
            rerank=rerank,
            candidate_multiplier=candidate_multiplier,
            max_steps=max_steps,
        )

        record_trace_step(
            db=db,
            run_id=run_id,
            step_index=0,
            step_type="agent_planning",
            name="rule_based_tool_selection",
            input_data={
                "query": cleaned_query,
                "available_tools": ["calculator", "mock_api", "document_search"],
            },
            output_data={
                "planned_tool_calls": plan,
                "plan_length": len(plan),
            },
            latency_ms=0,
        )

        tools = build_tool_registry(db=db)
        tool_calls: list[AgentToolCall] = []

        for index, planned_call in enumerate(plan[:max_steps], start=1):
            tool_name = planned_call["tool_name"]
            input_data = planned_call["input_data"]

            tool = tools.get(tool_name)

            if tool is None:
                result = ToolResult(
                    tool_name=tool_name,
                    input_data=input_data,
                    output_data={},
                    success=False,
                    error_message=f"Tool '{tool_name}' is not registered",
                )
            else:
                tool_start = time.perf_counter()
                result = tool.run(input_data)
                tool_latency_ms = elapsed_ms(tool_start)

                record_trace_step(
                    db=db,
                    run_id=run_id,
                    step_index=index,
                    step_type="tool_call",
                    name=tool_name,
                    input_data=result.input_data,
                    output_data=result.output_data,
                    status="completed" if result.success else "failed",
                    error_message=result.error_message,
                    latency_ms=tool_latency_ms,
                )

            tool_calls.append(
                AgentToolCall(
                    tool_name=result.tool_name,
                    input_data=result.input_data,
                    output_data=result.output_data,
                    success=result.success,
                    error_message=result.error_message,
                )
            )

        final_answer = build_final_answer(
            query=cleaned_query,
            tool_calls=tool_calls,
        )

        total_latency_ms = elapsed_ms(total_start)

        record_trace_step(
            db=db,
            run_id=run_id,
            step_index=len(tool_calls) + 1,
            step_type="agent_final_response",
            name="compose_final_answer",
            input_data={
                "query": cleaned_query,
                "tool_call_count": len(tool_calls),
            },
            output_data={
                "final_answer": final_answer,
            },
            latency_ms=0,
        )

        complete_run(
            db=db,
            run_id=run_id,
            output_answer=final_answer,
            latency_ms=total_latency_ms,
            metadata={
                "tool_call_count": len(tool_calls),
                "successful_tool_calls": sum(1 for call in tool_calls if call.success),
                "failed_tool_calls": sum(1 for call in tool_calls if not call.success),
            },
        )

        return AgentRunResult(
            run_id=run_id,
            query=cleaned_query,
            final_answer=final_answer,
            status="completed",
            tool_calls=tool_calls,
            total_latency_ms=total_latency_ms,
            metadata={
                "agent_version": "rule-based-agent-v1",
                "tool_call_count": len(tool_calls),
            },
        )

    except (RunRecorderError, AgentRunnerError) as exc:
        mark_agent_run_failed(
            db=db,
            run_id=run_id,
            error_message=str(exc),
            latency_ms=elapsed_ms(total_start),
        )
        raise AgentRunnerError(str(exc)) from exc

    except Exception as exc:
        logger.exception("Unexpected agent runner failure")
        mark_agent_run_failed(
            db=db,
            run_id=run_id,
            error_message="Failed to run tool-calling agent",
            latency_ms=elapsed_ms(total_start),
        )
        raise AgentRunnerError("Failed to run tool-calling agent") from exc

def build_agent_plan(
    query: str,
    document_id: Optional[str],
    retrieval_provider: str,
    top_k: int,
    rerank: bool,
    candidate_multiplier: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    normalized_query = query.lower()
    plan: list[dict[str, Any]] = []

    expression = extract_math_expression(query)

    if expression is not None:
        plan.append(
            {
                "tool_name": "calculator",
                "input_data": {
                    "expression": expression,
                },
            }
        )

    order_id = extract_pattern(query, r"\bORD-\d+\b")
    if order_id is not None:
        plan.append(
            {
                "tool_name": "mock_api",
                "input_data": {
                    "endpoint": "get_order_status",
                    "order_id": order_id,
                },
            }
        )

    customer_id = extract_pattern(query, r"\bCUST-\d+\b")
    if customer_id is not None:
        plan.append(
            {
                "tool_name": "mock_api",
                "input_data": {
                    "endpoint": "get_customer_plan",
                    "customer_id": customer_id,
                },
            }
        )

    if "weather" in normalized_query:
        city = extract_city(query)
        plan.append(
            {
                "tool_name": "mock_api",
                "input_data": {
                    "endpoint": "get_weather",
                    "city": city,
                },
            }
        )

    needs_document_search = (
        not plan
        or any(
            keyword in normalized_query
            for keyword in [
                "document",
                "policy",
                "refund",
                "return",
                "damaged",
                "missing",
                "according to",
                "provided documents",
            ]
        )
    )

    if needs_document_search:
        plan.append(
            {
                "tool_name": "document_search",
                "input_data": {
                    "query": query,
                    "document_id": document_id,
                    "retrieval_provider": retrieval_provider,
                    "top_k": top_k,
                    "rerank": rerank,
                    "candidate_multiplier": candidate_multiplier,
                },
            }
        )

    return plan[:max_steps]

def extract_math_expression(query: str) -> Optional[str]:
    expression_match = re.search(
        r"([-+*/().\d\s]+)",
        query,
    )

    if expression_match is None:
        return None

    expression = expression_match.group(1).strip()

    if not expression:
        return None

    if not any(operator in expression for operator in ["+", "-", "*", "/", "%"]):
        return None

    if not any(char.isdigit() for char in expression):
        return None

    return expression

def extract_pattern(query: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, query, flags=re.IGNORECASE)

    if match is None:
        return None

    return match.group(0).upper()

def extract_city(query: str) -> str:
    normalized_query = query.lower()

    for city in ["seattle", "sunnyvale", "hyderabad"]:
        if city in normalized_query:
            return city

    return "sunnyvale"

def build_final_answer(
    query: str,
    tool_calls: list[AgentToolCall],
) -> str:
    if not tool_calls:
        return "I could not select a suitable tool for this request."

    failed_calls = [call for call in tool_calls if not call.success]
    successful_calls = [call for call in tool_calls if call.success]

    if successful_calls:
        last_successful_call = successful_calls[-1]

        if last_successful_call.tool_name == "calculator":
            result_text = last_successful_call.output_data.get("result_text")
            return f"The calculated result is {result_text}."

        if last_successful_call.tool_name == "mock_api":
            return build_mock_api_answer(last_successful_call.output_data)

        if last_successful_call.tool_name == "document_search":
            chunks = last_successful_call.output_data.get("chunks", [])

            if not chunks:
                return (
                    "I could not find enough relevant evidence in the indexed "
                    "documents to answer this."
                )

            best_chunk = chunks[0]
            return (
                "I found relevant document evidence. "
                f"The strongest matching chunk is {best_chunk['chunk_id']} "
                f"with score {best_chunk['score']}."
            )

    if failed_calls:
        return failed_calls[-1].error_message or "The selected tool failed."

    return "The agent finished without a usable tool result."

def build_mock_api_answer(output_data: dict[str, Any]) -> str:
    if "order_id" in output_data:
        status = output_data.get("status")
        order_id = output_data.get("order_id")
        return f"Order {order_id} is currently {status}."

    if "customer_id" in output_data:
        customer_id = output_data.get("customer_id")
        plan = output_data.get("plan")
        sla = output_data.get("support_sla_hours")
        return f"Customer {customer_id} is on the {plan} plan with a {sla}-hour support SLA."

    if "city" in output_data:
        city = output_data.get("city")
        condition = output_data.get("condition")
        temperature = output_data.get("temperature_c")
        return f"The mock weather for {city} is {condition} at {temperature}°C."

    return str(output_data)

def elapsed_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)

def mark_agent_run_failed(
    db: Session,
    run_id: Optional[str],
    error_message: str,
    latency_ms: int,
) -> None:
    if run_id is None:
        return

    try:
        fail_run(
            db=db,
            run_id=run_id,
            error_message=error_message,
            latency_ms=latency_ms,
            metadata={
                "failure_stage": "agent_runner",
            },
        )
    except Exception:
        logger.exception("Failed to mark agent run as failed: %s", run_id)