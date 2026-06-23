from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.models import Document, DocumentChunk
from backend.app.database.schemas import DocumentChunkRead


router = APIRouter(
    prefix="/documents/{document_id}/chunks",
    tags=["Document Chunks"]
)


def get_document_or_404(document_id: str, db: Session) -> Document:
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' was not found"
        )

    return document


@router.get("", response_model=list[DocumentChunkRead])
def list_document_chunks(
    document_id: str,
    db: Annotated[Session, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50
):
    get_document_or_404(document_id, db)

    statement = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .offset(skip)
        .limit(limit)
    )

    chunks = db.execute(statement).scalars().all()

    return chunks


@router.get("/{chunk_id}", response_model=DocumentChunkRead)
def get_document_chunk(
    document_id: str,
    chunk_id: str,
    db: Annotated[Session, Depends(get_db)]
):
    get_document_or_404(document_id, db)

    statement = (
        select(DocumentChunk)
        .where(
            DocumentChunk.id == chunk_id,
            DocumentChunk.document_id == document_id
        )
    )

    chunk = db.execute(statement).scalars().first()

    if chunk is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk with id '{chunk_id}' was not found for document '{document_id}'"
        )

    return chunk