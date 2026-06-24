from dataclasses import dataclass
from pathlib import Path
from typing import Any
from sqlalchemy.orm import Session

from backend.app.database.models import Document, DocumentChunk
from backend.app.ingestion.document_loader import DocumentLoadError, load_document
from backend.app.ingestion.text_chunker import TextChunkingError, split_text_into_chunks
from backend.app.logging_config import get_logger

logger = get_logger(__name__)

class IngestionError(Exception):
    pass

@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    filename: str
    file_type: str
    status: str
    num_chunks: int
    metadata: dict[str, Any]

def ingest_document_file(
    file_path: str | Path,
    db: Session,
    chunk_size: int = 900,
    chunk_overlap: int = 150
) -> IngestionResult:
    try:
        loaded_document = load_document(file_path)

        chunks = split_text_into_chunks(
            loaded_document.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        document = Document(
            filename=loaded_document.filename,
            file_type=loaded_document.file_type,
            status="processing",
            num_pages=loaded_document.num_pages,
            num_chunks=0,
            metadata_json={
                **loaded_document.metadata,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        )

        db.add(document)
        db.flush()

        for chunk in chunks:
            document_chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.text,
                page_number=None,
                token_count=None,
                embedding_id=None,
                metadata_json=chunk.metadata
            )

            db.add(document_chunk)

        document.status = "chunked"
        document.num_chunks = len(chunks)

        db.commit()
        db.refresh(document)

        logger.info(
            "Ingested document %s with %s chunks",
            document.id,
            document.num_chunks
        )

        return IngestionResult(
            document_id=document.id,
            filename=document.filename,
            file_type=document.file_type,
            status=document.status,
            num_chunks=document.num_chunks,
            metadata=document.metadata_json
        )

    except (DocumentLoadError, TextChunkingError) as exc:
        db.rollback()
        raise IngestionError(str(exc)) from exc

    except Exception as exc:
        db.rollback()
        logger.exception("Unexpected ingestion failure")
        raise IngestionError("Failed to ingest document") from exc