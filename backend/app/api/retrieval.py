from typing import Any, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.retrieval.retrieval_service import (
    RetrievalError,
    retrieve_relevant_chunks,
)

router = APIRouter(
    prefix="/retrieval",
    tags=["Retrieval"]
)

class RetrievalSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(
        default=settings.default_retrieval_top_k,
        ge=1,
        le=settings.max_retrieval_top_k
    )
    document_id: Optional[str] = None

class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: dict[str, Any]

class RetrievalSearchResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunkResponse]
    top_k: int
    document_id: Optional[str]
    collection_name: str
    embedding_provider: str
    embedding_model: str

@router.post("/search", response_model=RetrievalSearchResponse)
def search_relevant_chunks(payload: RetrievalSearchRequest):
    try:
        result = retrieve_relevant_chunks(
            query=payload.query,
            top_k=payload.top_k,
            document_id=payload.document_id
        )

        return RetrievalSearchResponse(
            query=result.query,
            chunks=[
                RetrievedChunkResponse(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=chunk.score,
                    text=chunk.text,
                    metadata=chunk.metadata
                )
                for chunk in result.chunks
            ],
            top_k=result.top_k,
            document_id=result.document_id,
            collection_name=result.collection_name,
            embedding_provider=result.embedding_provider,
            embedding_model=result.embedding_model
        )

    except RetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc