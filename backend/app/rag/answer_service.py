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

MINIMUM_ANSWER_SCORE = 4.0

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "should", "would", "will", "i", "you", "we", "they",
    "he", "she", "it", "this", "that", "these", "those", "to", "for",
    "of", "in", "on", "at", "by", "with", "and", "or", "but", "from",
    "as", "get", "be", "what", "when", "where", "who", "whom", "whose",
    "why", "how", "tell", "me", "about", "please", "policy", "policies",
    "company", "customer", "customers", "apexcart", "offer", "accept"
}

METADATA_LINE_PREFIXES = (
    "version:",
    "effective date:",
    "document type:",
    "chunk_id:",
    "document_id:",
    "retrieval_score:",
)

REQUIREMENT_CUES = {
    "include", "includes", "included", "including",
    "require", "requires", "required", "requiring",
    "provide", "provides", "provided", "providing",
    "must", "need", "needs", "needed",
    "submit", "submits", "submitted",
}

EXCLUSION_CUES = {
    "exclude", "excludes", "excluded", "exclusion",
    "not cover", "does not cover", "do not cover", "doesn't cover",
    "not eligible", "non-refundable", "without", "unless",
}

TIME_CUES = {
    "day", "days", "business day", "business days",
    "calendar day", "calendar days", "hour", "hours",
    "month", "months", "year", "years",
    "billing cycle", "renewal date",
}

NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "first", "second", "third",
}

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

@dataclass(frozen=True)
class QuestionIntent:
    wants_requirement: bool
    wants_exclusion: bool
    wants_number: bool
    wants_time_or_duration: bool
    wants_multi_sentence_answer: bool

@dataclass(frozen=True)
class CandidateAnswer:
    text: str
    source_index: int
    sentence_index: int
    retrieval_score: float
    sentence_count: int

@dataclass(frozen=True)
class RankedCandidate:
    candidate: CandidateAnswer
    score: float
    matched_terms: set[str]
    reasons: list[str]

class AnswerGenerator(Protocol):
    generator_name: str

    def generate_answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        pass

class SimpleExtractiveAnswerGenerator:
    generator_name = "simple-extractive-v2"

    def generate_answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return SAFE_FALLBACK_ANSWER

        query_terms = extract_keywords(query)
        intent = detect_question_intent(query)

        candidate_answers = collect_candidate_answers(
            chunks=chunks,
            intent=intent
        )

        if not candidate_answers:
            return SAFE_FALLBACK_ANSWER

        ranked_candidates = rank_candidates(
            candidates=candidate_answers,
            query_terms=query_terms,
            intent=intent
        )

        best_answer = choose_best_candidate(ranked_candidates)

        if best_answer:
            return best_answer

        return SAFE_FALLBACK_ANSWER

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

def detect_question_intent(query: str) -> QuestionIntent:
    lowered_query = query.lower()

    wants_requirement = any(cue in lowered_query for cue in REQUIREMENT_CUES)
    wants_exclusion = any(cue in lowered_query for cue in EXCLUSION_CUES)

    wants_number = (
        "how many" in lowered_query
        or "number of" in lowered_query
    )

    wants_time_or_duration = (
        "how long" in lowered_query
        or "when" in lowered_query
        or "usual delivery time" in lowered_query
        or "delivery time" in lowered_query
        or "take effect" in lowered_query
        or "within" in lowered_query
    )

    wants_multi_sentence_answer = (
        wants_requirement
        or wants_exclusion
        or "what must" in lowered_query
        or "what should" in lowered_query
        or "what does" in lowered_query
    )

    return QuestionIntent(
        wants_requirement=wants_requirement,
        wants_exclusion=wants_exclusion,
        wants_number=wants_number,
        wants_time_or_duration=wants_time_or_duration,
        wants_multi_sentence_answer=wants_multi_sentence_answer
    )

def collect_candidate_answers(
    chunks: list[RetrievedChunk],
    intent: QuestionIntent
) -> list[CandidateAnswer]:
    candidates: list[CandidateAnswer] = []
    seen_candidates: set[str] = set()

    for source_index, chunk in enumerate(chunks):
        cleaned_text = clean_context_text(chunk.text)
        sentences = split_sentences(cleaned_text)

        for sentence_index, sentence in enumerate(sentences):
            add_candidate(
                candidates=candidates,
                seen_candidates=seen_candidates,
                text=sentence,
                source_index=source_index,
                sentence_index=sentence_index,
                retrieval_score=float(chunk.score),
                sentence_count=1
            )

            if not intent.wants_multi_sentence_answer:
                continue

            if sentence_index + 1 >= len(sentences):
                continue

            combined_text = f"{sentence} {sentences[sentence_index + 1]}"

            add_candidate(
                candidates=candidates,
                seen_candidates=seen_candidates,
                text=combined_text,
                source_index=source_index,
                sentence_index=sentence_index,
                retrieval_score=float(chunk.score),
                sentence_count=2
            )

    return candidates

def add_candidate(
    candidates: list[CandidateAnswer],
    seen_candidates: set[str],
    text: str,
    source_index: int,
    sentence_index: int,
    retrieval_score: float,
    sentence_count: int
) -> None:
    normalized_text = normalize_sentence(text)

    if not normalized_text:
        return

    candidate_key = normalized_text.lower()

    if candidate_key in seen_candidates:
        return

    seen_candidates.add(candidate_key)

    candidates.append(
        CandidateAnswer(
            text=normalized_text,
            source_index=source_index,
            sentence_index=sentence_index,
            retrieval_score=retrieval_score,
            sentence_count=sentence_count
        )
    )

def rank_candidates(
    candidates: list[CandidateAnswer],
    query_terms: set[str],
    intent: QuestionIntent
) -> list[RankedCandidate]:
    ranked_candidates: list[RankedCandidate] = []

    for candidate in candidates:
        candidate_terms = extract_keywords(candidate.text)
        matched_terms = query_terms.intersection(candidate_terms)

        score = 0.0
        reasons: list[str] = []

        if matched_terms:
            score += 2.0 * len(matched_terms)
            score += 3.0 * (len(matched_terms) / max(len(query_terms), 1))
            reasons.append("query_term_match")

        score += clamp(candidate.retrieval_score, 0.0, 1.0)

        lowered_text = candidate.text.lower()

        if intent.wants_requirement and contains_any(lowered_text, REQUIREMENT_CUES):
            score += 3.0
            reasons.append("requirement_cue")

        if intent.wants_exclusion and contains_exclusion_language(lowered_text):
            score += 5.0
            reasons.append("exclusion_cue")

        if intent.wants_number and contains_number_or_time_value(lowered_text):
            score += 3.0
            reasons.append("number_cue")

        if intent.wants_time_or_duration and contains_time_language(lowered_text):
            score += 3.0
            reasons.append("time_cue")

        if intent.wants_multi_sentence_answer and candidate.sentence_count > 1:
            if intent.wants_exclusion:
                score -= 0.5
                reasons.append("exclusion_multi_sentence_penalty")
            else:
                score += 0.75
                reasons.append("multi_sentence_context")

        if looks_like_overlap_fragment(candidate.text):
            score -= 4.0
            reasons.append("fragment_penalty")

        if looks_like_metadata_leak(candidate.text):
            score -= 5.0
            reasons.append("metadata_penalty")

        ranked_candidates.append(
            RankedCandidate(
                candidate=candidate,
                score=score,
                matched_terms=matched_terms,
                reasons=reasons
            )
        )

    return sorted(
        ranked_candidates,
        key=lambda item: (
            item.score,
            item.candidate.retrieval_score,
            -item.candidate.source_index,
            -item.candidate.sentence_index
        ),
        reverse=True
    )

def choose_best_candidate(
    ranked_candidates: list[RankedCandidate]
) -> Optional[str]:
    for ranked in ranked_candidates:
        if ranked.score < MINIMUM_ANSWER_SCORE:
            continue

        if not ranked.matched_terms:
            continue

        if looks_like_metadata_leak(ranked.candidate.text):
            continue

        return ranked.candidate.text

    return None

def extract_keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    keywords: set[str] = set()

    for token in tokens:
        if token in STOP_WORDS:
            continue

        normalized_token = normalize_keyword(token)

        if not normalized_token:
            continue

        if normalized_token in STOP_WORDS:
            continue

        keywords.add(normalized_token)

    return keywords

def normalize_keyword(token: str) -> str:
    synonym_map = {
        "cancellations": "cancellation",
        "subscriptions": "subscription",
        "payments": "payment",
        "charges": "charge",
        "days": "day",
        "hours": "hour",
        "times": "time",
        "usually": "usual",
        "reporting": "report",
        "reported": "report",
        "reports": "report",
        "damaged": "damage",
        "defective": "defect",
        "defects": "defect",
        "accessories": "accessory",
        "repairs": "repair",
        "repaired": "repair",
        "excludes": "cover",
        "excluded": "cover",
        "exclude": "cover",
        "exclusion": "cover",
        "coverage": "cover",
        "covered": "cover",
        "covering": "cover",
        "provides": "provide",
        "provided": "provide",
        "providing": "provide",
        "requires": "require",
        "required": "require",
        "requiring": "require",
        "includes": "include",
        "included": "include",
        "including": "include",
        "renewals": "renewal",
        "retries": "retry",
        "retrying": "retry",
        "addresses": "address",
        "invoices": "invoice",
        "records": "record",
        "plans": "plan",
    }

    if token in synonym_map:
        return synonym_map[token]

    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"

    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]

    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]

    if len(token) > 4 and token.endswith("ly"):
        return token[:-2]

    if (
        len(token) > 4
        and token.endswith("es")
        and token.endswith(("ches", "shes", "xes", "zes", "ses"))
    ):
        return token[:-2]

    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]

    return token

def clean_context_text(text: str) -> str:
    cleaned_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()

        if not stripped:
            continue

        if stripped.startswith("#"):
            continue

        if lowered.startswith(METADATA_LINE_PREFIXES):
            continue

        if re.match(r"^\[source\s+\d+\]$", lowered):
            continue

        cleaned_lines.append(stripped)

    return " ".join(cleaned_lines)

def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences: list[str] = []

    for part in parts:
        sentence = normalize_sentence(part)

        if sentence:
            sentences.append(sentence)

    return sentences

def normalize_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"^[-*•]\s+", "", sentence)
    sentence = re.sub(r"^\d+[.)]\s+", "", sentence)

    return sentence.strip()

def contains_any(text: str, cues: set[str]) -> bool:
    return any(cue in text for cue in cues)

def contains_exclusion_language(text: str) -> bool:
    return (
        "does not cover" in text
        or "do not cover" in text
        or "doesn't cover" in text
        or "not eligible" in text
        or "non-refundable" in text
        or "exclude" in text
        or "excluded" in text
        or "exclusion" in text
        or "unless" in text
        or "without" in text
    )

def contains_number_or_time_value(text: str) -> bool:
    if re.search(r"\b\d+\b", text):
        return True

    return any(word in text for word in NUMBER_WORDS)

def contains_time_language(text: str) -> bool:
    if contains_number_or_time_value(text) and contains_any(text, TIME_CUES):
        return True

    return any(cue in text for cue in TIME_CUES)

def looks_like_metadata_leak(sentence: str) -> bool:
    lowered = sentence.lower()

    return (
        "version:" in lowered
        or "effective date:" in lowered
        or "document type:" in lowered
        or "chunk_id:" in lowered
        or "document_id:" in lowered
        or "retrieval_score:" in lowered
    )

def looks_like_overlap_fragment(sentence: str) -> bool:
    words = sentence.split()

    if len(words) <= 3:
        return True

    first_character = sentence[0]

    if first_character.islower():
        return True

    return False

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))