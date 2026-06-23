from dataclasses import dataclass
from typing import Any, Optional

from backend.app.config import settings
from backend.app.embeddings.embedding_provider import (
    EmbeddingError,
    EmbeddingProvider,
    HashEmbeddingProvider,
)
from backend.app.logging_config import get_logger
from backend.app.vector_store.qdrant_store import QdrantStore, VectorStoreError


logger = get_logger(__name__)

class RetrievalError(Exception):
    pass

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: dict[str, Any]

@dataclass(frozen=True)
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    top_k: int
    document_id: Optional[str]
    collection_name: str
    embedding_provider: str
    embedding_model: str

def retrieve_relevant_chunks(
    query: str,
    top_k: int = settings.default_retrieval_top_k,
    document_id: Optional[str] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    vector_store: Optional[QdrantStore] = None
) -> RetrievalResult:
    cleaned_query = query.strip()

    if not cleaned_query:
        raise RetrievalError("Query cannot be empty")

    if top_k <= 0:
        raise RetrievalError("top_k must be greater than 0")

    if top_k > settings.max_retrieval_top_k:
        raise RetrievalError(
            f"top_k cannot be greater than {settings.max_retrieval_top_k}"
        )

    provider = embedding_provider or HashEmbeddingProvider()
    store = vector_store or QdrantStore(vector_size=provider.dimension)

    try:
        query_embedding = provider.embed_text(cleaned_query)

        search_results = store.search(
            query_vector=query_embedding.vector,
            limit=top_k,
            document_id=document_id
        )

        chunks = [
            RetrievedChunk(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                score=result.score,
                text=result.text,
                metadata=result.payload.get("metadata", {})
            )
            for result in search_results
        ]

        logger.info(
            "Retrieved %s chunks for query: %s",
            len(chunks),
            cleaned_query
        )

        return RetrievalResult(
            query=cleaned_query,
            chunks=chunks,
            top_k=top_k,
            document_id=document_id,
            collection_name=store.collection_name,
            embedding_provider=provider.provider_name,
            embedding_model=provider.model_name
        )

    except (EmbeddingError, VectorStoreError) as exc:
        raise RetrievalError(str(exc)) from exc

    except Exception as exc:
        logger.exception("Unexpected retrieval failure")
        raise RetrievalError("Failed to retrieve relevant chunks") from exc