from typing import Annotated, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.agents.agent_runner import AgentRunnerError, run_tool_calling_agent
from backend.app.config import settings
from backend.app.database.connection import get_db


router = APIRouter(
    prefix="/agents",
    tags=["Agents"],
)

class AgentRunRequest(BaseModel):
    query: str = Field(..., min_length=1)
    document_id: Optional[str] = None
    retrieval_provider: str = "hybrid"
    top_k: int = Field(
        default=settings.default_retrieval_top_k,
        ge=1,
        le=settings.max_retrieval_top_k,
    )
    rerank: bool = True
    candidate_multiplier: int = Field(default=3, ge=1, le=10)
    max_steps: int = Field(default=5, ge=1, le=10)

class AgentToolCallResponse(BaseModel):
    tool_name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    success: bool
    error_message: Optional[str]

class AgentRunResponse(BaseModel):
    run_id: str
    query: str
    final_answer: str
    status: str
    tool_calls: list[AgentToolCallResponse]
    total_latency_ms: int
    metadata: dict[str, Any]

@router.post("/run", response_model=AgentRunResponse)
def run_agent(
    payload: AgentRunRequest,
    db: Annotated[Session, Depends(get_db)],
):
    try:
        result = run_tool_calling_agent(
            db=db,
            query=payload.query,
            document_id=payload.document_id,
            retrieval_provider=payload.retrieval_provider,
            top_k=payload.top_k,
            rerank=payload.rerank,
            candidate_multiplier=payload.candidate_multiplier,
            max_steps=payload.max_steps,
        )

        return AgentRunResponse(
            run_id=result.run_id,
            query=result.query,
            final_answer=result.final_answer,
            status=result.status,
            tool_calls=[
                AgentToolCallResponse(
                    tool_name=tool_call.tool_name,
                    input_data=tool_call.input_data,
                    output_data=tool_call.output_data,
                    success=tool_call.success,
                    error_message=tool_call.error_message,
                )
                for tool_call in result.tool_calls
            ],
            total_latency_ms=result.total_latency_ms,
            metadata=result.metadata,
        )

    except AgentRunnerError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc