import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from backend.app.config import settings
from backend.app.logging_config import get_logger
from backend.app.retrieval.retrieval_service import (
    RetrievalError,
    RetrievedChunk,
    retrieve_relevant_chunks,
)

logger = get_logger(__name__)

SAFE_FALLBACK_ANSWER = (
    "I could not find enough reliable evidence in the provided documents "
    "to answer this confidently."
)

class RAGAnswerError(Exception):
    pass

@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: dict[str, Any]

@dataclass(frozen=True)
class RAGAnswerResult:
    query: str
    answer: str
    source_chunks: list[SourceChunk]
    retrieval_top_k: int
    document_id: Optional[str]
    answer_generator: str

class AnswerGenerator(Protocol):
    generator_name: str

    def generate_answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        pass

class SimpleExtractiveAnswerGenerator:
    generator_name = "simple-extractive-v1"

    def generate_answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return SAFE_FALLBACK_ANSWER

        query_terms = extract_keywords(query)
        candidate_sentences = collect_candidate_sentences(chunks)

        if not candidate_sentences:
            return "I found relevant chunks, but they did not contain a clear sentence-level answer."

        ranked_sentences = rank_sentences(candidate_sentences, query_terms)
        best_sentence = choose_best_sentence(ranked_sentences)

        if best_sentence:
            return best_sentence

        return (
            "I could not find a direct answer in the retrieved context. "
            "The closest retrieved context is: "
            f"{clean_context_text(chunks[0].text)}"
        )

def answer_question(
    query: str,
    top_k: int = settings.default_retrieval_top_k,
    document_id: Optional[str] = None,
    answer_generator: Optional[AnswerGenerator] = None
) -> RAGAnswerResult:
    cleaned_query = query.strip()

    if not cleaned_query:
        raise RAGAnswerError("Query cannot be empty")

    generator = answer_generator or SimpleExtractiveAnswerGenerator()

    try:
        retrieval_result = retrieve_relevant_chunks(
            query=cleaned_query,
            top_k=top_k,
            document_id=document_id
        )

        answer = generator.generate_answer(
            query=cleaned_query,
            chunks=retrieval_result.chunks
        )

        source_chunks = [
            SourceChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                score=chunk.score,
                text=chunk.text,
                metadata=chunk.metadata
            )
            for chunk in retrieval_result.chunks
        ]

        logger.info("Generated grounded answer for query: %s", cleaned_query)

        return RAGAnswerResult(
            query=cleaned_query,
            answer=answer,
            source_chunks=source_chunks,
            retrieval_top_k=top_k,
            document_id=document_id,
            answer_generator=generator.generator_name
        )

    except RetrievalError as exc:
        raise RAGAnswerError(str(exc)) from exc

    except Exception as exc:
        logger.exception("Unexpected RAG answer failure")
        raise RAGAnswerError("Failed to answer question") from exc

def extract_keywords(text: str) -> set[str]:
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
        "can", "could", "should", "would", "i", "you", "we", "they", "he",
        "she", "it", "this", "that", "these", "those", "to", "for", "of",
        "in", "on", "at", "by", "with", "and", "or", "but", "from", "as",
        "get", "be"
    }

    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return {token for token in tokens if token not in stop_words}

def clean_context_text(text: str) -> str:
    cleaned_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            continue

        cleaned_lines.append(stripped)

    return " ".join(cleaned_lines)

def collect_candidate_sentences(chunks: list[RetrievedChunk]) -> list[str]:
    sentences: list[str] = []
    seen_sentences: set[str] = set()

    for chunk in chunks:
        cleaned_text = clean_context_text(chunk.text)
        parts = re.split(r"(?<=[.!?])\s+", cleaned_text)

        for part in parts:
            sentence = normalize_sentence(part)

            if not sentence:
                continue
            sentence_key = sentence.lower()
            if sentence_key in seen_sentences:
                continue

            seen_sentences.add(sentence_key)
            sentences.append(sentence)

    return sentences

def normalize_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"^[-*•]\s+", "", sentence)
    sentence = re.sub(r"^\d+[.)]\s+", "", sentence)

    return sentence.strip()

def rank_sentences(
    sentences: list[str],
    query_terms: set[str]
) -> list[tuple[str, int, int]]:
    ranked: list[tuple[str, int, int]] = []

    for index, sentence in enumerate(sentences):
        sentence_terms = extract_keywords(sentence)
        score = len(query_terms.intersection(sentence_terms))

        ranked.append((sentence, score, index))

    return sorted(
        ranked,
        key=lambda item: (item[1], -item[2]),
        reverse=True
    )

def choose_best_sentence(
    ranked_sentences: list[tuple[str, int, int]]
) -> Optional[str]:
    for sentence, score, _ in ranked_sentences:
        if score <= 0:
            continue
        if looks_like_overlap_fragment(sentence):
            continue
        return sentence
    
    for sentence, score, _ in ranked_sentences:
        if score > 0:
            return sentence

    return None

def looks_like_overlap_fragment(sentence: str) -> bool:
    words = sentence.split()

    if len(words) <= 3:
        return True

    first_character = sentence[0]

    if first_character.islower():
        return True

    return False