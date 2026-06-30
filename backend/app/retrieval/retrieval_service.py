from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Optional

from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.embeddings.embedding_provider import (
    EmbeddingError,
    EmbeddingProvider,
    get_default_embedding_provider,
)
from backend.app.logging_config import get_logger
from backend.app.retrieval.bm25_retriever import BM25ChunkScore, retrieve_bm25_chunks
from backend.app.retrieval.reranker import rerank_chunks
from backend.app.vector_store.qdrant_store import QdrantStore, VectorStoreError

logger = get_logger(__name__)
RetrievalMethod = Literal["dense", "bm25", "hybrid"]
SUPPORTED_RETRIEVAL_METHODS = {"dense", "bm25", "hybrid"}

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
    retrieval_method: str
    reranker_used: bool = False
    reranker_name: Optional[str] = None

def retrieve_relevant_chunks(
    query: str,
    top_k: int = settings.default_retrieval_top_k,
    document_id: Optional[str] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    vector_store: Optional[QdrantStore] = None,
    rerank: bool = False,
    candidate_multiplier: int = 3,
) -> RetrievalResult:
    return retrieve_chunks(
        query=query,
        top_k=top_k,
        document_id=document_id,
        method="dense",
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        rerank=rerank,
        candidate_multiplier=candidate_multiplier,
    )

def retrieve_chunks(
    query: str,
    top_k: int = settings.default_retrieval_top_k,
    document_id: Optional[str] = None,
    method: RetrievalMethod = "dense",
    db: Optional[Session] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    vector_store: Optional[QdrantStore] = None,
    rerank: bool = False,
    candidate_multiplier: int = 3,
) -> RetrievalResult:
    cleaned_query = validate_retrieval_request(query=query, top_k=top_k, method=method)
    candidate_top_k = resolve_candidate_top_k(
        top_k=top_k,
        rerank=rerank,
        candidate_multiplier=candidate_multiplier,
    )

    if method == "dense":
        result = retrieve_dense_chunks(
            query=cleaned_query,
            top_k=candidate_top_k,
            document_id=document_id,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
        return apply_optional_reranking(
            result=result,
            query=cleaned_query,
            final_top_k=top_k,
            rerank=rerank,
        )

    if db is None:
        raise RetrievalError(f"Database session is required for {method} retrieval")

    if method == "bm25":
        result = retrieve_bm25_result(
            db=db,
            query=cleaned_query,
            top_k=candidate_top_k,
            document_id=document_id,
        )
        return apply_optional_reranking(
            result=result,
            query=cleaned_query,
            final_top_k=top_k,
            rerank=rerank,
        )

    result = retrieve_hybrid_chunks(
        db=db,
        query=cleaned_query,
        top_k=candidate_top_k,
        document_id=document_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    return apply_optional_reranking(
        result=result,
        query=cleaned_query,
        final_top_k=top_k,
        rerank=rerank,
    )

def compare_retrieval_methods(
    query: str,
    methods: list[RetrievalMethod],
    top_k: int = settings.default_retrieval_top_k,
    document_id: Optional[str] = None,
    db: Optional[Session] = None,
    rerank: bool = False,
    candidate_multiplier: int  = 3,
) -> list[RetrievalResult]:
    if not methods:
        raise RetrievalError("At least one retrieval method is required")
    results = []
    for method in dict.fromkeys(methods):
        results.append(
            retrieve_chunks(
                query=query,
                top_k=top_k,
                document_id=document_id,
                method=method,
                db=db,
                rerank=rerank,
                candidate_multiplier=candidate_multiplier,
            )
        )
    return results

def resolve_candidate_top_k(
    top_k: int,
    rerank: bool,
    candidate_multiplier: int,
) -> int:
    if not rerank:
        return top_k

    if candidate_multiplier <= 0:
        raise RetrievalError("candidate_multiplier must be greater than 0")

    return min(
        settings.max_retrieval_top_k,
        max(top_k, top_k * candidate_multiplier),
    )

def apply_optional_reranking(
    result: RetrievalResult,
    query: str,
    final_top_k: int,
    rerank: bool,
) -> RetrievalResult:
    if not rerank:
        return result

    reranked_items = rerank_chunks(
        query=query,
        chunks=result.chunks,
    )

    reranked_chunks = []

    for rank, item in enumerate(reranked_items[:final_top_k], start=1):
        chunk = item.chunk
        metadata = dict(chunk.metadata)
        actual_reranker_name = item.details.get("reranker_name")

        metadata["reranker_used"] = True
        metadata["reranker_name"] = actual_reranker_name
        metadata["rerank_score"] = item.score
        metadata["rerank_rank"] = rank
        metadata["rerank_details"] = item.details

        reranked_chunks.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                score=item.score,
                text=chunk.text,
                metadata=metadata,
            )
        )

    result_reranker_name = (
        reranked_items[0].details.get("reranker_name")
        if reranked_items
        else None
    )

    return replace(
        result,
        chunks=reranked_chunks,
        top_k=final_top_k,
        reranker_used=True,
        reranker_name=result_reranker_name,
    )

def validate_retrieval_request(query: str, top_k: int, method: str) -> str:
    cleaned_query = query.strip()
    if not cleaned_query:
        raise RetrievalError("Query cannot be empty")
    if top_k <= 0:
        raise RetrievalError("top_k must be greater than 0")
    if top_k > settings.max_retrieval_top_k:
        raise RetrievalError(f"top_k cannot be greater than {settings.max_retrieval_top_k}")
    if method not in SUPPORTED_RETRIEVAL_METHODS:
        supported = ", ".join(sorted(SUPPORTED_RETRIEVAL_METHODS))
        raise RetrievalError(f"Unsupported retrieval method '{method}'. Supported methods: {supported}")
    return cleaned_query

def retrieve_dense_chunks(
    query: str,
    top_k: int,
    document_id: Optional[str] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    vector_store: Optional[QdrantStore] = None,
) -> RetrievalResult:
    provider = embedding_provider or get_default_embedding_provider()
    store = vector_store or QdrantStore(vector_size=provider.dimension)
    try:
        query_embedding = provider.embed_text(query)
        search_results = store.search(
            query_vector=query_embedding.vector,
            limit=top_k,
            document_id=document_id,
        )
        chunks = [
            RetrievedChunk(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                score=result.score,
                text=result.text,
                metadata={
                    **(result.payload.get("metadata", {}) or {}),
                    "retrieval_method": "dense",
                    "dense_score": result.score,
                },
            )
            for result in search_results
        ]
        logger.info("Dense retrieval returned %s chunks for query: %s", len(chunks), query)
        return RetrievalResult(
            query=query,
            chunks=chunks,
            top_k=top_k,
            document_id=document_id,
            collection_name=store.collection_name,
            embedding_provider=provider.provider_name,
            embedding_model=provider.model_name,
            retrieval_method="dense",
        )
    except (EmbeddingError, VectorStoreError) as exc:
        raise RetrievalError(str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected dense retrieval failure")
        raise RetrievalError("Failed to retrieve relevant chunks") from exc

def retrieve_bm25_result(
    db: Session,
    query: str,
    top_k: int,
    document_id: Optional[str] = None,
) -> RetrievalResult:
    try:
        bm25_chunks = retrieve_bm25_chunks(
            db=db,
            query=query,
            top_k=top_k,
            document_id=document_id,
        )
        chunks = [chunk_from_bm25_score(chunk) for chunk in bm25_chunks]
        logger.info("BM25 retrieval returned %s chunks for query: %s", len(chunks), query)
        return RetrievalResult(
            query=query,
            chunks=chunks,
            top_k=top_k,
            document_id=document_id,
            collection_name="postgres_document_chunks",
            embedding_provider="none",
            embedding_model="none",
            retrieval_method="bm25",
        )
    except Exception as exc:
        logger.exception("Unexpected BM25 retrieval failure")
        raise RetrievalError("Failed to retrieve BM25 chunks") from exc

def retrieve_hybrid_chunks(
    db: Session,
    query: str,
    top_k: int,
    document_id: Optional[str] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    vector_store: Optional[QdrantStore] = None,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
) -> RetrievalResult:
    if dense_weight < 0 or bm25_weight < 0:
        raise RetrievalError("Hybrid retrieval weights must be non-negative")
    if dense_weight + bm25_weight <= 0:
        raise RetrievalError("At least one hybrid retrieval weight must be positive")
    candidate_limit = min(settings.max_retrieval_top_k, max(top_k * 3, top_k))
    dense_result = retrieve_dense_chunks(
        query=query,
        top_k=candidate_limit,
        document_id=document_id,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    bm25_chunks = retrieve_bm25_chunks(
        db=db,
        query=query,
        top_k=candidate_limit,
        document_id=document_id,
    )
    dense_scores = normalize_scores({chunk.chunk_id: chunk.score for chunk in dense_result.chunks})
    bm25_scores = normalize_scores({chunk.chunk_id: chunk.score for chunk in bm25_chunks})
    combined: dict[str, dict[str, Any]] = {}
    for chunk in dense_result.chunks:
        combined[chunk.chunk_id] = {
            "chunk": chunk,
            "dense_raw": chunk.score,
            "dense_score": dense_scores.get(chunk.chunk_id, 0.0),
            "bm25_raw": 0.0,
            "bm25_score": 0.0,
        }
    for bm25_chunk in bm25_chunks:
        if bm25_chunk.chunk_id not in combined:
            combined[bm25_chunk.chunk_id] = {
                "chunk": chunk_from_bm25_score(bm25_chunk),
                "dense_raw": 0.0,
                "dense_score": 0.0,
                "bm25_raw": 0.0,
                "bm25_score": 0.0,
            }
        combined[bm25_chunk.chunk_id]["bm25_raw"] = bm25_chunk.score
        combined[bm25_chunk.chunk_id]["bm25_score"] = bm25_scores.get(bm25_chunk.chunk_id, 0.0)
    ranked_chunks = []
    for item in combined.values():
        chunk = item["chunk"]
        score = dense_weight * item["dense_score"] + bm25_weight * item["bm25_score"]
        metadata = dict(chunk.metadata)
        metadata["retrieval_method"] = "hybrid"
        metadata["retrieval_scores"] = {
            "dense_raw": item["dense_raw"],
            "dense_score": item["dense_score"],
            "bm25_raw": item["bm25_raw"],
            "bm25_score": item["bm25_score"],
            "dense_weight": dense_weight,
            "bm25_weight": bm25_weight,
        }
        ranked_chunks.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                score=round(score, 6),
                text=chunk.text,
                metadata=metadata,
            )
        )
    ranked_chunks.sort(key=lambda chunk: chunk.score, reverse=True)
    logger.info("Hybrid retrieval returned %s chunks for query: %s", min(len(ranked_chunks), top_k), query)
    return RetrievalResult(
        query=query,
        chunks=ranked_chunks[:top_k],
        top_k=top_k,
        document_id=document_id,
        collection_name="hybrid:qdrant_dense+postgres_bm25",
        embedding_provider=dense_result.embedding_provider,
        embedding_model=dense_result.embedding_model,
        retrieval_method="hybrid",
    )

def chunk_from_bm25_score(chunk: BM25ChunkScore) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        score=chunk.score,
        text=chunk.text,
        metadata={
            **chunk.metadata,
            "retrieval_method": "bm25",
            "bm25_score": chunk.score,
        },
    )

def normalize_scores(scores_by_id: dict[str, float]) -> dict[str, float]:
    if not scores_by_id:
        return {}
    values = list(scores_by_id.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {item_id: 1.0 for item_id in scores_by_id}
    return {
        item_id: round((score - min_score) / (max_score - min_score), 6)
        for item_id, score in scores_by_id.items()
    }