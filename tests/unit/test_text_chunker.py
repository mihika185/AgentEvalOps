import pytest

from backend.app.ingestion.text_chunker import (
    TextChunkingError,
    split_text_into_chunks,
)


def test_short_text_creates_single_chunk():
    text = "Customers can request a refund within 7 days."

    chunks = split_text_into_chunks(
        text=text,
        chunk_size=900,
        chunk_overlap=150
    )

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata["start_char"] == 0
    assert chunks[0].metadata["end_char"] == len(text)
    assert chunks[0].metadata["character_count"] == len(text)
    assert chunks[0].metadata["split_strategy"] == "boundary_aware"


def test_long_text_creates_multiple_chunks():
    text = " ".join(
        f"Sentence {index} about refund policy."
        for index in range(100)
    )

    chunks = split_text_into_chunks(
        text=text,
        chunk_size=300,
        chunk_overlap=50
    )

    assert len(chunks) > 1

    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index
        assert chunk.text.strip()

        start_char = chunk.metadata["start_char"]
        end_char = chunk.metadata["end_char"]

        assert start_char < end_char
        assert end_char <= len(text)
        assert chunk.metadata["character_count"] == len(chunk.text)


def test_chunks_have_overlap_when_text_is_long():
    text = " ".join(
        f"Policy sentence number {index}."
        for index in range(80)
    )

    chunks = split_text_into_chunks(
        text=text,
        chunk_size=250,
        chunk_overlap=40
    )

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:]):
        previous_end = previous.metadata["end_char"]
        current_start = current.metadata["start_char"]

        assert current_start < previous_end


def test_empty_text_raises_chunking_error():
    with pytest.raises(TextChunkingError, match="Cannot chunk empty text"):
        split_text_into_chunks(
            text="   ",
            chunk_size=900,
            chunk_overlap=150
        )


def test_invalid_chunk_size_raises_error():
    with pytest.raises(TextChunkingError, match="chunk_size must be greater than 0"):
        split_text_into_chunks(
            text="Valid text",
            chunk_size=0,
            chunk_overlap=0
        )


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(TextChunkingError, match="chunk_overlap must be smaller"):
        split_text_into_chunks(
            text="Valid text",
            chunk_size=100,
            chunk_overlap=100
        )