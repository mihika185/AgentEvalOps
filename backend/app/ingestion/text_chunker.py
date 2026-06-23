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
    paragraphs: list[str] = []
    current_paragraph: list[str] = []

    for line in text.splitlines():
        cleaned_line = " ".join(line.strip().split())

        if cleaned_line:
            current_paragraph.append(cleaned_line)
            continue

        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))
            current_paragraph = []

    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    return "\n\n".join(paragraphs)

def find_chunk_end(text: str, start: int, chunk_size: int) -> int:
    text_length = len(text)
    max_end = min(start + chunk_size, text_length)

    if max_end == text_length:
        return text_length

    min_end = start + max(1, chunk_size // 2)

    boundary_markers = [
        "\n\n",
        ". ",
        "? ",
        "! ",
        "\n",
        "; ",
        ": ",
        ", ",
        " "
    ]

    for marker in boundary_markers:
        boundary_index = text.rfind(marker, min_end, max_end)

        if boundary_index != -1:
            return boundary_index + len(marker)

    return max_end

def move_to_word_start(text: str, start: int) -> int:
    if start <= 0:
        return 0

    if start >= len(text):
        return len(text)

    if text[start].isspace():
        while start < len(text) and text[start].isspace():
            start += 1

        return start

    if text[start - 1].isspace():
        return start

    while start > 0 and not text[start - 1].isspace():
        start -= 1

    return start

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
        end = find_chunk_end(cleaned_text, start, chunk_size)

        if end <= start:
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
                        "character_count": len(chunk_text),
                        "split_strategy": "boundary_aware"
                    }
                )
            )

        if end == text_length:
            break

        next_start = max(0, end - chunk_overlap)
        next_start = move_to_word_start(cleaned_text, next_start)

        if next_start <= start:
            next_start = end

        start = next_start

    logger.info("Split text into %s chunks", len(chunks))

    return chunks