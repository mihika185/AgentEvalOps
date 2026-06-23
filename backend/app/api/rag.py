from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.rag.answer_service import RAGAnswerError, answer_question


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


class RAGAnswerResponse(BaseModel):
    query: str
    answer: str
    source_chunks: list[SourceChunkResponse]
    retrieval_top_k: int
    document_id: Optional[str]
    answer_generator: str


@router.post("/answer", response_model=RAGAnswerResponse)
def generate_rag_answer(payload: RAGAnswerRequest):
    try:
        result = answer_question(
            query=payload.query,
            top_k=payload.top_k,
            document_id=payload.document_id
        )

        return RAGAnswerResponse(
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
            answer_generator=result.answer_generator
        )

    except RAGAnswerError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc