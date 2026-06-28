from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Any, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import DocumentChunk

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")

@dataclass(frozen=True)
class BM25CandidateChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    metadata: dict[str, Any]

@dataclass(frozen=True)
class BM25ChunkScore:
    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: dict[str, Any]

def tokenize_text(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text or "")
    ]

def score_chunks_with_bm25(
    query: str,
    chunks: list[BM25CandidateChunk],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[BM25ChunkScore]:
    query_terms = sorted(set(tokenize_text(query)))

    if not query_terms or not chunks:
        return []

    tokenized_chunks = [
        (chunk, tokenize_text(chunk.text))
        for chunk in chunks
    ]

    document_count = len(tokenized_chunks)
    average_doc_length = (
        sum(len(tokens) for _, tokens in tokenized_chunks) / document_count
    )

    if average_doc_length <= 0:
        average_doc_length = 1.0

    document_frequencies = {
        term: 0
        for term in query_terms
    }

    for _, tokens in tokenized_chunks:
        token_set = set(tokens)

        for term in query_terms:
            if term in token_set:
                document_frequencies[term] += 1

    scored_chunks: list[tuple[int, BM25ChunkScore]] = []

    for chunk, tokens in tokenized_chunks:
        if not tokens:
            continue

        term_counts: dict[str, int] = {}

        for token in tokens:
            if token in document_frequencies:
                term_counts[token] = term_counts.get(token, 0) + 1

        score = 0.0
        document_length = len(tokens)

        for term in query_terms:
            term_frequency = term_counts.get(term, 0)

            if term_frequency == 0:
                continue

            doc_frequency = document_frequencies[term]

            idf = math.log(
                1.0
                + (
                    (document_count - doc_frequency + 0.5)
                    / (doc_frequency + 0.5)
                )
            )

            denominator = term_frequency + (
                k1
                * (
                    1.0
                    - b
                    + b * (document_length / average_doc_length)
                )
            )

            score += idf * ((term_frequency * (k1 + 1.0)) / denominator)

        if score <= 0:
            continue

        scored_chunks.append(
            (
                chunk.chunk_index,
                BM25ChunkScore(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=round(score, 6),
                    text=chunk.text,
                    metadata=chunk.metadata,
                ),
            )
        )

    scored_chunks.sort(
        key=lambda item: (-item[1].score, item[0])
    )

    return [
        score
        for _, score in scored_chunks
    ]

def retrieve_bm25_chunks(
    db: Session,
    query: str,
    top_k: int,
    document_id: Optional[str] = None,
) -> list[BM25ChunkScore]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    statement = select(DocumentChunk)

    if document_id is not None:
        statement = statement.where(DocumentChunk.document_id == document_id)

    statement = statement.order_by(
        DocumentChunk.document_id.asc(),
        DocumentChunk.chunk_index.asc(),
    )

    rows = db.execute(statement).scalars().all()

    candidate_chunks = [
        BM25CandidateChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            chunk_index=row.chunk_index,
            text=row.chunk_text,
            metadata=row.metadata_json or {},
        )
        for row in rows
    ]

    return score_chunks_with_bm25(
        query=query,
        chunks=candidate_chunks,
    )[:top_k]