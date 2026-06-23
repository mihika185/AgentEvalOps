from typing import Annotated, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database.connection import get_db
from backend.app.rag.workflow_service import (
    RAGWorkflowError,
    run_rag_answer_workflow,
)

router = APIRouter(
    prefix="/rag",
    tags=["RAG"]
)

class RAGAnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(
        default=settings.default_retrieval_top_k,
        ge=1,
        le=settings.max_retrieval_top_k
    )
    document_id: Optional[str] = None

class SourceChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: dict[str, Any]

class EvaluationMetricResponse(BaseModel):
    metric_name: str
    metric_value: float
    details: dict[str, Any]

class RAGAnswerResponse(BaseModel):
    run_id: str
    query: str
    answer: str
    source_chunks: list[SourceChunkResponse]
    retrieval_top_k: int
    document_id: Optional[str]
    answer_generator: str
    total_latency_ms: int
    evaluation_metrics: list[EvaluationMetricResponse]
    quality_gate_passed: bool
    quality_gate_pass_rate: float
    failed_quality_gates: list[str]

@router.post("/answer", response_model=RAGAnswerResponse)
def generate_rag_answer(
    payload: RAGAnswerRequest,
    db: Annotated[Session, Depends(get_db)]
):
    try:
        result = run_rag_answer_workflow(
            db=db,
            query=payload.query,
            top_k=payload.top_k,
            document_id=payload.document_id
        )

        return RAGAnswerResponse(
            run_id=result.run_id,
            query=result.query,
            answer=result.answer,
            source_chunks=[
                SourceChunkResponse(
                    chunk_id=source.chunk_id,
                    document_id=source.document_id,
                    score=source.score,
                    text=source.text,
                    metadata=source.metadata
                )
                for source in result.source_chunks
            ],
            retrieval_top_k=result.retrieval_top_k,
            document_id=result.document_id,
            answer_generator=result.answer_generator,
            total_latency_ms=result.total_latency_ms,
            evaluation_metrics=[
                EvaluationMetricResponse(
                    metric_name=metric.metric_name,
                    metric_value=metric.metric_value,
                    details=metric.details
                )
                for metric in result.evaluation_metrics
            ],
            quality_gate_passed=result.quality_gate_passed,
            quality_gate_pass_rate=result.quality_gate_pass_rate,
            failed_quality_gates=result.failed_quality_gates
        )

    except RAGWorkflowError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc