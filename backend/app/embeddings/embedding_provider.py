import hashlib
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Protocol


class EmbeddingError(Exception):
    pass


@dataclass(frozen=True)
class EmbeddingResult:
    text: str
    vector: list[float]
    provider: str
    model: str
    dimension: int


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    dimension: int

    def embed_text(self, text: str) -> EmbeddingResult:
        ...

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        ...


class HashEmbeddingProvider:
    provider_name = "hash"
    model_name = "hash-embedding-v1"

    def __init__(self, dimension: int = 384):
        if dimension <= 0:
            raise EmbeddingError("Embedding dimension must be greater than 0")

        self.dimension = dimension

    def embed_text(self, text: str) -> EmbeddingResult:
        tokens = tokenize_text(text)

        if not tokens:
            raise EmbeddingError("Cannot embed empty text")

        vector = [0.0] * self.dimension

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()

            index = int.from_bytes(digest[:4], byteorder="big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0

            vector[index] += sign

        normalized_vector = normalize_vector(vector)

        return EmbeddingResult(
            text=text,
            vector=normalized_vector,
            provider=self.provider_name,
            model=self.model_name,
            dimension=self.dimension
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.embed_text(text) for text in texts]


class SentenceTransformerEmbeddingProvider:
    provider_name = "sentence-transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Install it with: python3 -m pip install "
                "\"sentence-transformers>=3.0.0,<4.0.0\""
            ) from exc

        self.model = SentenceTransformer(model_name)

        dimension = self.model.get_sentence_embedding_dimension()

        if dimension is None:
            sample_vector = self.model.encode(
                "dimension probe",
                normalize_embeddings=True
            )
            dimension = len(sample_vector)

        self.dimension = int(dimension)

    def embed_text(self, text: str) -> EmbeddingResult:
        cleaned_text = text.strip()

        if not cleaned_text:
            raise EmbeddingError("Cannot embed empty text")

        vector = self.model.encode(
            cleaned_text,
            normalize_embeddings=True
        )

        vector_list = [float(value) for value in vector.tolist()]

        return EmbeddingResult(
            text=text,
            vector=vector_list,
            provider=self.provider_name,
            model=self.model_name,
            dimension=len(vector_list)
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []

        cleaned_texts = []

        for text in texts:
            cleaned_text = text.strip()

            if not cleaned_text:
                raise EmbeddingError("Cannot embed empty text")

            cleaned_texts.append(cleaned_text)

        vectors = self.model.encode(
            cleaned_texts,
            normalize_embeddings=True
        )

        results: list[EmbeddingResult] = []

        for original_text, vector in zip(texts, vectors):
            vector_list = [float(value) for value in vector.tolist()]

            results.append(
                EmbeddingResult(
                    text=original_text,
                    vector=vector_list,
                    provider=self.provider_name,
                    model=self.model_name,
                    dimension=len(vector_list)
                )
            )

        return results


@lru_cache(maxsize=4)
def get_embedding_provider(
    provider_name: str,
    model_name: Optional[str] = None
) -> EmbeddingProvider:
    provider_key = provider_name.strip().lower()

    if provider_key in {"hash", "local"}:
        return HashEmbeddingProvider()

    if provider_key in {"sentence-transformers", "sentence_transformers", "sbert"}:
        return SentenceTransformerEmbeddingProvider(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2"
        )

    raise EmbeddingError(f"Unsupported embedding provider: {provider_name}")


def get_default_embedding_provider() -> EmbeddingProvider:
    from backend.app.config import settings

    return get_embedding_provider(
        provider_name=settings.default_embedding_provider,
        model_name=settings.default_embedding_model
    )


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))

    if norm == 0:
        return vector

    return [value / norm for value in vector]


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise EmbeddingError("Vectors must have the same dimension")

    return sum(a * b for a, b in zip(vector_a, vector_b))