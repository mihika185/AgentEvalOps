import uuid
from dataclasses import dataclass
from typing import Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from backend.app.config import settings
from backend.app.embeddings.embedding_provider import EmbeddingResult
from backend.app.logging_config import get_logger


logger = get_logger(__name__)

DEFAULT_COLLECTION_NAME = "agentevalops_chunks"


class VectorStoreError(Exception):
    pass


@dataclass(frozen=True)
class ChunkEmbedding:
    chunk_id: str
    document_id: str
    text: str
    embedding: EmbeddingResult
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: str
    document_id: str
    score: float
    text: str
    payload: dict[str, Any]


class QdrantStore:
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = 384,
        url: str = settings.qdrant_url
    ):
        if vector_size <= 0:
            raise VectorStoreError("vector_size must be greater than 0")

        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(url=url)

    def ensure_collection(self) -> None:
        if self.client.collection_exists(collection_name=self.collection_name):
            logger.info("Qdrant collection already exists: %s", self.collection_name)
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )

        logger.info("Created Qdrant collection: %s", self.collection_name)

    def reset_collection(self) -> None:
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )

        logger.info("Reset Qdrant collection: %s", self.collection_name)

    def upsert_chunk_embeddings(self, chunk_embeddings: list[ChunkEmbedding]) -> int:
        if not chunk_embeddings:
            return 0

        self.ensure_collection()

        points: list[PointStruct] = []

        for item in chunk_embeddings:
            self._validate_embedding_dimension(item.embedding)

            payload = {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "text": item.text,
                "provider": item.embedding.provider,
                "model": item.embedding.model,
                "metadata": item.metadata
            }

            points.append(
                PointStruct(
                    id=self._point_id_from_chunk_id(item.chunk_id),
                    vector=item.embedding.vector,
                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )

        logger.info(
            "Upserted %s vectors into Qdrant collection %s",
            len(points),
            self.collection_name
        )

        return len(points)

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        document_id: Optional[str] = None
    ) -> list[VectorSearchResult]:
        if len(query_vector) != self.vector_size:
            raise VectorStoreError(
                f"Query vector dimension {len(query_vector)} does not match "
                f"collection vector size {self.vector_size}"
            )

        self.ensure_collection()

        query_filter = None

        if document_id is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )

        results: list[VectorSearchResult] = []

        for point in response.points:
            payload = point.payload or {}

            results.append(
                VectorSearchResult(
                    chunk_id=str(payload.get("chunk_id", "")),
                    document_id=str(payload.get("document_id", "")),
                    score=float(point.score),
                    text=str(payload.get("text", "")),
                    payload=payload
                )
            )

        return results

    def delete_document_vectors(self, document_id: str) -> None:
        self.ensure_collection()

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            ),
            wait=True
        )

        logger.info("Deleted Qdrant vectors for document: %s", document_id)

    def _validate_embedding_dimension(self, embedding: EmbeddingResult) -> None:
        if embedding.dimension != self.vector_size:
            raise VectorStoreError(
                f"Embedding dimension {embedding.dimension} does not match "
                f"collection vector size {self.vector_size}"
            )

        if len(embedding.vector) != self.vector_size:
            raise VectorStoreError(
                f"Embedding vector length {len(embedding.vector)} does not match "
                f"collection vector size {self.vector_size}"
            )

    @staticmethod
    def _point_id_from_chunk_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agentevalops:{chunk_id}"))