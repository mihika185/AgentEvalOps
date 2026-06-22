from dataclasses import dataclass
from typing import Any

from backend.app.logging_config import get_logger

logger = get_logger(__name__)

class TextChunkingError(Exception):
    pass

@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    text: str
    metadata: dict[str, Any]

    @property
    def character_count(self) -> int:
        return len(self.text)

def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line]

    return "\n".join(cleaned_lines)

def split_text_into_chunks(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise TextChunkingError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise TextChunkingError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise TextChunkingError("chunk_overlap must be smaller than chunk_size")

    cleaned_text = normalize_text(text)

    if not cleaned_text:
        raise TextChunkingError("Cannot chunk empty text")

    chunks: list[TextChunk] = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunk_text = cleaned_text[start:end].strip()

        if chunk_text:
            chunks.append(
                TextChunk(
                    chunk_index=len(chunks),
                    text=chunk_text,
                    metadata={
                        "start_char": start,
                        "end_char": end,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "character_count": len(chunk_text)
                    }
                )
            )

        if end == text_length:
            break

        start = end - chunk_overlap

    logger.info("Split text into %s chunks", len(chunks))

    return chunks