import tempfile
from pathlib import Path
from typing import Annotated
from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.models import Document
from backend.app.database.schemas import DocumentCreate, DocumentRead, DocumentUpdate
from backend.app.ingestion.ingestion_service import IngestionError, ingest_document_file
from backend.app.indexing.indexing_service import IndexingError, index_document_chunks
from backend.app.logging_config import get_logger

router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)

class UploadedIndexedDocumentResponse(DocumentRead):
    indexed_chunks: int
    vector_collection: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int

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

@router.post(
    "/upload-and-index",
    response_model=UploadedIndexedDocumentResponse,
    status_code=status.HTTP_201_CREATED
)
async def upload_and_index_document(
    db: Annotated[Session, Depends(get_db)],
    file: UploadFile = File(...)
):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename"
        )

    suffix = Path(file.filename).suffix.lower()

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            content = await file.read()
            temp_file.write(content)

        ingestion_result = ingest_document_file(
            file_path=temp_path,
            db=db
        )

        document = get_document_or_404(
            document_id=ingestion_result.document_id,
            db=db
        )

        original_source_path = document.metadata_json.get("source_path")

        document.filename = file.filename
        document.metadata_json = {
            **(document.metadata_json or {}),
            "source_path": file.filename,
            "original_filename": file.filename,
            "upload_source": "api_upload",
            "temporary_source_path": original_source_path
        }

        db.commit()
        db.refresh(document)

        indexing_result = index_document_chunks(
            document_id=ingestion_result.document_id,
            db=db
        )

        document = get_document_or_404(
            document_id=ingestion_result.document_id,
            db=db
        )

        return UploadedIndexedDocumentResponse(
            id=document.id,
            filename=file.filename,
            file_type=document.file_type,
            status=document.status,
            num_pages=document.num_pages,
            num_chunks=document.num_chunks,
            metadata_json=document.metadata_json,
            created_at=document.created_at,
            updated_at=document.updated_at,
            indexed_chunks=indexing_result.indexed_chunks,
            vector_collection=indexing_result.collection_name,
            embedding_provider=indexing_result.embedding_provider,
            embedding_model=indexing_result.embedding_model,
            embedding_dimension=indexing_result.embedding_dimension
        )

    except IngestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc

    except IndexingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        ) from exc

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

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