import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

from backend.app.logging_config import get_logger

logger = get_logger(__name__)

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
        pass

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        pass

class HashEmbeddingProvider:
    provider_name = "local"
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