from typing import Annotated, Any, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database.connection import get_db
from backend.app.evaluation.quality_gates import DEFAULT_QUALITY_GATE_PROFILE
from backend.app.rag.workflow_service import RAGWorkflowError, run_rag_answer_workflow

router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)

RetrievalProvider = Literal["dense", "bm25", "hybrid"]

class RAGAnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(
        default=settings.default_retrieval_top_k,
        ge=1,
        le=settings.max_retrieval_top_k,
    )
    rerank: bool = False
    candidate_multiplier: int = Field(default=3, ge=1, le=10)
    document_id: Optional[str] = None
    retrieval_provider: RetrievalProvider = "dense"
    quality_gate_profile: str = Field(
        default=DEFAULT_QUALITY_GATE_PROFILE,
        min_length=1,
    )

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
    retrieval_provider: str
    source_chunks: list[SourceChunkResponse]
    retrieval_top_k: int
    retrieved_chunk_count: int
    document_id: Optional[str]
    answer_generator: str
    reranker_used: bool
    reranker_name: Optional[str]
    total_latency_ms: int
    evaluation_metrics: list[EvaluationMetricResponse]
    quality_gate_profile: str
    quality_gate_passed: bool
    quality_gate_pass_rate: float
    failed_quality_gates: list[str]
    response_blocked_by_quality_gate: bool

@router.post("/answer", response_model=RAGAnswerResponse)
def generate_rag_answer(
    payload: RAGAnswerRequest,
    db: Annotated[Session, Depends(get_db)],
):
    try:
        result = run_rag_answer_workflow(
            db=db,
            query=payload.query,
            top_k=payload.top_k,
            document_id=payload.document_id,
            retrieval_provider=payload.retrieval_provider,
            quality_gate_profile=payload.quality_gate_profile,
            rerank=payload.rerank,
            candidate_multiplier=payload.candidate_multiplier,
        )
        return RAGAnswerResponse(
            run_id=result.run_id,
            query=result.query,
            answer=result.answer,
            retrieval_provider=result.retrieval_provider,
            source_chunks=[
                SourceChunkResponse(
                    chunk_id=source.chunk_id,
                    document_id=source.document_id,
                    score=source.score,
                    text=source.text,
                    metadata=source.metadata,
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
                    details=metric.details,
                )
                for metric in result.evaluation_metrics
            ],
            quality_gate_profile=result.quality_gate_profile,
            quality_gate_passed=result.quality_gate_passed,
            quality_gate_pass_rate=result.quality_gate_pass_rate,
            failed_quality_gates=result.failed_quality_gates,
            retrieved_chunk_count=result.retrieved_chunk_count,
            reranker_used=result.reranker_used,
            reranker_name=result.reranker_name,
            response_blocked_by_quality_gate=result.response_blocked_by_quality_gate,
        )
    except RAGWorkflowError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc