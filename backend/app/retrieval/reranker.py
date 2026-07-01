from __future__ import annotations
import re

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Literal, Optional

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CROSS_ENCODER_RERANKER_NAME = "cross_encoder_v1"
TERM_OVERLAP_RERANKER_NAME = "term_overlap_fallback_v1"

RERANKER_NAME = CROSS_ENCODER_RERANKER_NAME

RerankerProvider = Literal["cross_encoder", "term_overlap"]
CrossEncoderScorer = Callable[[list[tuple[str, str]], str], list[float]]

@dataclass(frozen=True)
class RerankResult:
    chunk: Any
    score: float
    details: dict[str, Any]

class RerankingError(Exception):
    pass

def tokenize_text(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text or "")
    ]

def unique_terms(text: str) -> set[str]:
    return set(tokenize_text(text))

def rerank_chunks(
    query: str,
    chunks: list[Any],
    provider: RerankerProvider = "cross_encoder",
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    fallback_to_term_overlap: bool = True,
    cross_encoder_scorer: Optional[CrossEncoderScorer] = None,
    min_cross_encoder_score: Optional[float] = 0.0,
) -> list[RerankResult]:
    if not chunks:
        return []

    cleaned_query = query.strip()

    if not cleaned_query:
        return rerank_by_term_overlap(query=query, chunks=chunks)

    if provider == "term_overlap":
        return rerank_by_term_overlap(query=cleaned_query, chunks=chunks)

    if provider != "cross_encoder":
        raise RerankingError(f"Unsupported reranker provider: {provider}")

    try:
        return rerank_by_cross_encoder(
            query=cleaned_query,
            chunks=chunks,
            model_name=model_name,
            cross_encoder_scorer=cross_encoder_scorer,
            min_cross_encoder_score=min_cross_encoder_score,
        )

    except Exception as exc:
        if not fallback_to_term_overlap:
            raise RerankingError(f"Cross-encoder reranking failed: {exc}") from exc

        fallback_results = rerank_by_term_overlap(
            query=cleaned_query,
            chunks=chunks,
        )

        return [
            RerankResult(
                chunk=item.chunk,
                score=item.score,
                details={
                    **item.details,
                    "fallback_from": CROSS_ENCODER_RERANKER_NAME,
                    "fallback_reason": str(exc),
                },
            )
            for item in fallback_results
        ]

def rerank_by_cross_encoder(
    query: str,
    chunks: list[Any],
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    cross_encoder_scorer: Optional[CrossEncoderScorer] = None,
    min_cross_encoder_score: Optional[float] = 0.0,
) -> list[RerankResult]:
    pairs = [
        (query, getattr(chunk, "text", "") or "")
        for chunk in chunks
    ]

    if cross_encoder_scorer is not None:
        raw_scores = cross_encoder_scorer(pairs, model_name)
    else:
        raw_scores = score_cross_encoder_pairs(pairs, model_name)

    if len(raw_scores) != len(chunks):
        raise RerankingError(
            "Cross-encoder scorer returned a different number of scores "
            "than the number of chunks"
        )

    normalized_scores = normalize_float_scores(raw_scores)
    base_scores = normalize_base_scores(chunks)

    results = []

    for chunk, raw_score, normalized_score in zip(
        chunks,
        raw_scores,
        normalized_scores,
    ):
        chunk_id = str(chunk.chunk_id)

        results.append(
            RerankResult(
                chunk=chunk,
                score=normalized_score,
                details={
                    "reranker_name": CROSS_ENCODER_RERANKER_NAME,
                    "reranker_provider": "cross_encoder",
                    "reranker_model": model_name,
                    "raw_cross_encoder_score": round(float(raw_score), 6),
                    "normalized_cross_encoder_score": normalized_score,
                    "base_score": base_scores.get(chunk_id, 0.0),
                },
            )
        )

    results.sort(
        key=lambda item: (
            -item.score,
            -item.details["base_score"],
            str(item.chunk.chunk_id),
        )
    )

    if min_cross_encoder_score is None:
        return results

    filtered_results = [
        item
        for item in results
        if item.details["raw_cross_encoder_score"] >= min_cross_encoder_score
    ]

    if filtered_results:
        return [
            RerankResult(
                chunk=item.chunk,
                score=item.score,
                details={
                    **item.details,
                    "min_cross_encoder_score": min_cross_encoder_score,
                    "filtered_by_cross_encoder_score": True,
                },
            )
            for item in filtered_results
        ]

    best_result = results[0]

    return [
        RerankResult(
            chunk=best_result.chunk,
            score=best_result.score,
            details={
                **best_result.details,
                "min_cross_encoder_score": min_cross_encoder_score,
                "filtered_by_cross_encoder_score": True,
                "kept_as_best_available_result": True,
            },
        )
    ]

def score_cross_encoder_pairs(
    pairs: list[tuple[str, str]],
    model_name: str,
) -> list[float]:
    model = get_cross_encoder_model(model_name)
    scores = model.predict(pairs)

    return [
        float(score)
        for score in scores
    ]

@lru_cache(maxsize=2)
def get_cross_encoder_model(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)

def rerank_by_term_overlap(
    query: str,
    chunks: list[Any],
    base_weight: float = 0.35,
    overlap_weight: float = 0.5,
    phrase_weight: float = 0.15,
) -> list[RerankResult]:
    query_terms = unique_terms(query)
    base_scores = normalize_base_scores(chunks)
    results = []

    for chunk in chunks:
        chunk_id = str(chunk.chunk_id)
        text_terms = unique_terms(getattr(chunk, "text", ""))
        matched_terms = sorted(query_terms & text_terms)

        overlap_score = (
            len(matched_terms) / len(query_terms)
            if query_terms
            else 0.0
        )

        phrase_score = phrase_match_score(
            query=query,
            text=getattr(chunk, "text", ""),
        )

        base_score = base_scores.get(chunk_id, 0.0)

        rerank_score = (
            base_weight * base_score
            + overlap_weight * overlap_score
            + phrase_weight * phrase_score
        )

        results.append(
            RerankResult(
                chunk=chunk,
                score=round(rerank_score, 6),
                details={
                    "reranker_name": TERM_OVERLAP_RERANKER_NAME,
                    "reranker_provider": "term_overlap",
                    "base_score": base_score,
                    "term_overlap_score": round(overlap_score, 6),
                    "phrase_match_score": phrase_score,
                    "matched_terms": matched_terms,
                },
            )
        )

    results.sort(
        key=lambda item: (
            -item.score,
            -item.details["base_score"],
            str(item.chunk.chunk_id),
        )
    )

    return results

def normalize_base_scores(chunks: list[Any]) -> dict[str, float]:
    if not chunks:
        return {}

    scores = {
        str(chunk.chunk_id): float(chunk.score)
        for chunk in chunks
    }

    min_score = min(scores.values())
    max_score = max(scores.values())

    if max_score == min_score:
        return {
            chunk_id: 1.0
            for chunk_id in scores
        }

    return {
        chunk_id: round((score - min_score) / (max_score - min_score), 6)
        for chunk_id, score in scores.items()
    }

def normalize_float_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0 for _ in scores]

    return [
        round((score - min_score) / (max_score - min_score), 6)
        for score in scores
    ]

def phrase_match_score(query: str, text: str) -> float:
    cleaned_query = " ".join(tokenize_text(query))
    cleaned_text = " ".join(tokenize_text(text))

    if not cleaned_query or not cleaned_text:
        return 0.0

    return 1.0 if cleaned_query in cleaned_text else 0.0