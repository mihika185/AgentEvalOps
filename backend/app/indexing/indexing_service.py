from dataclasses import dataclass
from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import Document, DocumentChunk
from backend.app.embeddings.embedding_provider import (
    EmbeddingError,
    EmbeddingProvider,
    HashEmbeddingProvider,
)
from backend.app.logging_config import get_logger
from backend.app.vector_store.qdrant_store import (
    ChunkEmbedding,
    QdrantStore,
    VectorStoreError,
)

logger = get_logger(__name__)

class IndexingError(Exception):
    pass

@dataclass(frozen=True)
class IndexingResult:
    document_id: str
    status: str
    indexed_chunks: int
    collection_name: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int

def index_document_chunks(
    document_id: str,
    db: Session,
    embedding_provider: Optional[EmbeddingProvider] = None,
    vector_store: Optional[QdrantStore] = None,
    batch_size: int = 64
) -> IndexingResult:
    if batch_size <= 0:
        raise IndexingError("batch_size must be greater than 0")

    provider = embedding_provider or HashEmbeddingProvider()
    store = vector_store or QdrantStore(vector_size=provider.dimension)

    document = db.get(Document, document_id)

    if document is None:
        raise IndexingError(f"Document with id '{document_id}' was not found")

    chunks = db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
    ).scalars().all()

    if not chunks:
        raise IndexingError(f"Document '{document_id}' has no chunks to index")

    try:
        document.status = "indexing"
        db.flush()

        store.delete_document_vectors(document_id)

        indexed_count = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [chunk.chunk_text for chunk in batch]
            embeddings = provider.embed_batch(texts)

            chunk_embeddings: list[ChunkEmbedding] = []

            for chunk, embedding in zip(batch, embeddings):
                metadata = {
                    **(chunk.metadata_json or {}),
                    "chunk_index": chunk.chunk_index,
                    "filename": document.filename,
                    "file_type": document.file_type,
                }

                chunk_embeddings.append(
                    ChunkEmbedding(
                        chunk_id=chunk.id,
                        document_id=document.id,
                        text=chunk.chunk_text,
                        embedding=embedding,
                        metadata=metadata
                    )
                )

            store.upsert_chunk_embeddings(chunk_embeddings)

            for chunk in batch:
                chunk.embedding_id = f"{store.collection_name}:{chunk.id}"

            indexed_count += len(batch)

        document.status = "indexed"
        document.metadata_json = {
            **(document.metadata_json or {}),
            "embedding_provider": provider.provider_name,
            "embedding_model": provider.model_name,
            "embedding_dimension": provider.dimension,
            "vector_collection": store.collection_name,
        }

        db.commit()
        db.refresh(document)

        logger.info(
            "Indexed document %s with %s chunks",
            document.id,
            indexed_count
        )

        return IndexingResult(
            document_id=document.id,
            status=document.status,
            indexed_chunks=indexed_count,
            collection_name=store.collection_name,
            embedding_provider=provider.provider_name,
            embedding_model=provider.model_name,
            embedding_dimension=provider.dimension
        )

    except (EmbeddingError, VectorStoreError) as exc:
        db.rollback()
        raise IndexingError(str(exc)) from exc

    except Exception as exc:
        db.rollback()
        logger.exception("Unexpected indexing failure")
        raise IndexingError("Failed to index document") from exc