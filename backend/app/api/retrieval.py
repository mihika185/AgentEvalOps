from typing import Any, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database.connection import get_db
from backend.app.retrieval.retrieval_service import (
    RetrievalError,
    RetrievalResult,
    compare_retrieval_methods,
    retrieve_chunks,
)

router = APIRouter(
    prefix="/retrieval",
    tags=["Retrieval"],
)

RetrievalMethod = Literal["dense", "bm25", "hybrid"]

class RetrievalSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    method: RetrievalMethod = "dense"
    top_k: int = Field(
        default=settings.default_retrieval_top_k,
        ge=1,
        le=settings.max_retrieval_top_k,
    )
    document_id: Optional[str] = None

class RetrievalCompareRequest(BaseModel):
    query: str = Field(..., min_length=1)
    methods: list[RetrievalMethod] = Field(
        default_factory=lambda: ["bm25", "dense", "hybrid"],
        min_length=1,
    )
    top_k: int = Field(
        default=settings.default_retrieval_top_k,
        ge=1,
        le=settings.max_retrieval_top_k,
    )
    document_id: Optional[str] = None

    @field_validator("methods")
    @classmethod
    def remove_duplicate_methods(cls, value: list[RetrievalMethod]) -> list[RetrievalMethod]:
        return list(dict.fromkeys(value))

class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: dict[str, Any]

class RetrievalSearchResponse(BaseModel):
    query: str
    method: str
    chunks: list[RetrievedChunkResponse]
    top_k: int
    document_id: Optional[str]
    collection_name: str
    embedding_provider: str
    embedding_model: str

class RetrievalCompareResponse(BaseModel):
    query: str
    top_k: int
    document_id: Optional[str]
    results: list[RetrievalSearchResponse]

@router.post("/search", response_model=RetrievalSearchResponse)
def search_relevant_chunks(
    payload: RetrievalSearchRequest,
    db: Session = Depends(get_db),
):
    try:
        result = retrieve_chunks(
            query=payload.query,
            top_k=payload.top_k,
            document_id=payload.document_id,
            method=payload.method,
            db=db,
        )
        return to_retrieval_search_response(result)
    except RetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

@router.post("/compare", response_model=RetrievalCompareResponse)
def compare_retrievers(
    payload: RetrievalCompareRequest,
    db: Session = Depends(get_db),
):
    try:
        results = compare_retrieval_methods(
            query=payload.query,
            methods=payload.methods,
            top_k=payload.top_k,
            document_id=payload.document_id,
            db=db,
        )
        return RetrievalCompareResponse(
            query=payload.query.strip(),
            top_k=payload.top_k,
            document_id=payload.document_id,
            results=[to_retrieval_search_response(result) for result in results],
        )
    except RetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

def to_retrieval_search_response(result: RetrievalResult) -> RetrievalSearchResponse:
    return RetrievalSearchResponse(
        query=result.query,
        method=result.retrieval_method,
        chunks=[
            RetrievedChunkResponse(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                score=chunk.score,
                text=chunk.text,
                metadata=chunk.metadata,
            )
            for chunk in result.chunks
        ],
        top_k=result.top_k,
        document_id=result.document_id,
        collection_name=result.collection_name,
        embedding_provider=result.embedding_provider,
        embedding_model=result.embedding_model,
    )