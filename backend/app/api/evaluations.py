from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database.connection import get_db
from backend.app.evaluation.agent_metrics import (
    AgentEvaluationError,
    evaluate_agent_run,
)
from backend.app.evaluation.eval_runner import EvalRunnerError, run_agent_eval_set
from backend.app.experiments.experiment_service import (
    ExperimentServiceError,
    ensure_experiment_exists,
)


router = APIRouter(
    prefix="/evaluations",
    tags=["Evaluations"],
)

class AgentEvalSetRunRequest(BaseModel):
    eval_set_name: str = Field(default="agent_eval_set.json", min_length=1)
    document_id: Optional[str] = None
    retrieval_provider: str = "hybrid"
    top_k: int = Field(default=3, ge=1, le=settings.max_retrieval_top_k)
    rerank: bool = True
    candidate_multiplier: int = Field(default=3, ge=1, le=10)
    max_steps: int = Field(default=5, ge=1, le=10)
    experiment_id: Optional[str] = None

class AgentRunEvaluationRequest(BaseModel):
    expected_tools: list[str] = Field(default_factory=list)
    expected_answer_terms: list[str] = Field(default_factory=list)
    max_allowed_tool_calls: Optional[int] = Field(default=None, ge=1)

class EvaluationMetricResponse(BaseModel):
    metric_name: str
    metric_value: float
    details: dict[str, Any]

class AgentEvalCaseResultResponse(BaseModel):
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

class AgentEvalSetRunResponse(BaseModel):
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
    results: list[AgentEvalCaseResultResponse]

class AgentRunEvaluationResponse(BaseModel):
    run_id: str
    evaluator_type: str
    passed: bool
    failure_reasons: list[str]
    metrics: list[EvaluationMetricResponse]

@router.post(
    "/agent-eval-sets/run",
    response_model=AgentEvalSetRunResponse,
)
def run_agent_eval_set_endpoint(
    payload: AgentEvalSetRunRequest,
    db: Annotated[Session, Depends(get_db)],
):
    try:
        eval_set_path = resolve_eval_set_path(payload.eval_set_name)
        experiment_id = ensure_experiment_exists(db, payload.experiment_id)

        summary = run_agent_eval_set(
            db=db,
            eval_set_path=eval_set_path,
            document_id=payload.document_id,
            retrieval_provider=payload.retrieval_provider,
            top_k=payload.top_k,
            rerank=payload.rerank,
            candidate_multiplier=payload.candidate_multiplier,
            max_steps=payload.max_steps,
            experiment_id=experiment_id,
        )

        return AgentEvalSetRunResponse(
            eval_set_path=summary.eval_set_path,
            total_cases=summary.total_cases,
            passed_cases=summary.passed_cases,
            failed_cases=summary.failed_cases,
            pass_rate=summary.pass_rate,
            average_tool_selection_accuracy=summary.average_tool_selection_accuracy,
            average_tool_order_accuracy=summary.average_tool_order_accuracy,
            average_answer_correctness=summary.average_answer_correctness,
            average_tool_call_success_rate=summary.average_tool_call_success_rate,
            average_unnecessary_tool_call_rate=summary.average_unnecessary_tool_call_rate,
            average_overall_agent_score=summary.average_overall_agent_score,
            results=[
                AgentEvalCaseResultResponse(
                    case_id=result.case_id,
                    question=result.question,
                    run_id=result.run_id,
                    passed=result.passed,
                    failure_reason=result.failure_reason,
                    metrics=result.metrics,
                    expected_tools=result.expected_tools,
                    actual_tools=result.actual_tools,
                    final_answer=result.final_answer,
                    latency_ms=result.latency_ms,
                )
                for result in summary.results
            ],
        )

    except ExperimentServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except EvalRunnerError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

@router.post(
    "/agent-runs/{run_id}",
    response_model=AgentRunEvaluationResponse,
)
def evaluate_existing_agent_run(
    run_id: str,
    payload: AgentRunEvaluationRequest,
    db: Annotated[Session, Depends(get_db)],
):
    try:
        summary = evaluate_agent_run(
            db=db,
            run_id=run_id,
            expected_tools=payload.expected_tools,
            expected_answer_terms=payload.expected_answer_terms,
            max_allowed_tool_calls=payload.max_allowed_tool_calls,
            persist=True,
        )

        return AgentRunEvaluationResponse(
            run_id=summary.run_id,
            evaluator_type=summary.evaluator_type,
            passed=summary.passed,
            failure_reasons=summary.failure_reasons,
            metrics=[
                EvaluationMetricResponse(
                    metric_name=metric.metric_name,
                    metric_value=metric.metric_value,
                    details=metric.details,
                )
                for metric in summary.metrics
            ],
        )

    except AgentEvaluationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

def resolve_eval_set_path(eval_set_name: str) -> str:
    cleaned_name = eval_set_name.strip()

    if not cleaned_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="eval_set_name cannot be empty",
        )

    if "/" in cleaned_name or "\\" in cleaned_name or ".." in cleaned_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="eval_set_name must be a file name inside data/eval_sets",
        )

    return f"data/eval_sets/{cleaned_name}"