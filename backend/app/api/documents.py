from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.models import Document
from backend.app.database.schemas import DocumentCreate, DocumentRead, DocumentUpdate
from backend.app.logging_config import get_logger

router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)

logger = get_logger(__name__)

def get_document_or_404(document_id: str, db: Session) -> Document:
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' was not found"
        )

    return document

@router.post("", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
def create_document(
    payload: DocumentCreate,
    db: Annotated[Session, Depends(get_db)]
):
    document = Document(
        filename=payload.filename,
        file_type=payload.file_type,
        metadata_json=payload.metadata_json
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    logger.info("Created document record: %s", document.id)

    return document

@router.get("", response_model=list[DocumentRead])
def list_documents(
    db: Annotated[Session, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20
):
    statement = (
        select(Document)
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
    )

    documents = db.execute(statement).scalars().all()

    return documents

@router.get("/{document_id}", response_model=DocumentRead)
def get_document(
    document_id: str,
    db: Annotated[Session, Depends(get_db)]
):
    return get_document_or_404(document_id, db)

@router.patch("/{document_id}", response_model=DocumentRead)
def update_document(
    document_id: str,
    payload: DocumentUpdate,
    db: Annotated[Session, Depends(get_db)]
):
    document = get_document_or_404(document_id, db)

    update_data = payload.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(document, field, value)

    db.commit()
    db.refresh(document)

    logger.info("Updated document record: %s", document.id)

    return document

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    db: Annotated[Session, Depends(get_db)]
):
    document = get_document_or_404(document_id, db)

    db.delete(document)
    db.commit()

    logger.info("Deleted document record: %s", document_id)

    return Response(status_code=status.HTTP_204_NO_CONTENT)