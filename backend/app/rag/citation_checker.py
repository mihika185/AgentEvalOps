import re
from dataclasses import dataclass
from typing import Any

from backend.app.rag.answer_service import SAFE_FALLBACK_ANSWER, SourceChunk


MINIMUM_CITATION_SUPPORT_SCORE = 0.20

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "should", "would", "will", "i", "you", "we", "they",
    "he", "she", "it", "this", "that", "these", "those", "to", "for",
    "of", "in", "on", "at", "by", "with", "and", "or", "but", "from",
    "as", "get", "be", "what", "when", "where", "who", "whom", "whose",
    "why", "how", "tell", "me", "about", "please", "policy", "policies",
    "company", "customer", "customers", "may", "must", "does", "using",
    "provided", "documents", "document", "enough", "reliable", "evidence",
    "confidently", "answer", "answers", "question",
}

@dataclass(frozen=True)
class Citation:
    source_number: int
    chunk_id: str
    document_id: str
    retrieval_score: float
    support_score: float
    matched_terms: list[str]
    text_excerpt: str
    metadata: dict[str, Any]

@dataclass(frozen=True)
class CitationCheckResult:
    citations: list[Citation]
    citation_check_passed: bool
    citation_accuracy_score: float
    failed_reasons: list[str]
    cited_chunk_ids: list[str]
    valid_citation_count: int
    total_citation_count: int

def check_answer_citations(
    answer: str,
    source_chunks: list[SourceChunk],
    max_citations: int = 3,
)->CitationCheckResult:
    cleaned_answer = normalize_text(answer)

    if not cleaned_answer:
        return CitationCheckResult(
            citations=[],
            citation_check_passed=False,
            citation_accuracy_score=0.0,
            failed_reasons=["answer_is_empty"],
            cited_chunk_ids=[],
            valid_citation_count=0,
            total_citation_count=0,
        )

    if is_fallback_answer(cleaned_answer):
        return CitationCheckResult(
            citations=[],
            citation_check_passed=False,
            citation_accuracy_score=0.0,
            failed_reasons=["answer_is_safe_fallback"],
            cited_chunk_ids=[],
            valid_citation_count=0,
            total_citation_count=0,
        )

    if not source_chunks:
        return CitationCheckResult(
            citations=[],
            citation_check_passed=False,
            citation_accuracy_score=0.0,
            failed_reasons=["no_source_chunks_available"],
            cited_chunk_ids=[],
            valid_citation_count=0,
            total_citation_count=0,
        )

    citations = build_citations(
        answer=cleaned_answer,
        source_chunks=source_chunks,
        max_citations=max_citations,
    )

    if not citations:
        return CitationCheckResult(
            citations=[],
            citation_check_passed=False,
            citation_accuracy_score=0.0,
            failed_reasons=["no_source_chunk_supported_answer_terms"],
            cited_chunk_ids=[],
            valid_citation_count=0,
            total_citation_count=0,
        )

    retrieved_chunk_ids = {
        chunk.chunk_id
        for chunk in source_chunks
    }

    invalid_citations = [
        citation
        for citation in citations
        if citation.chunk_id not in retrieved_chunk_ids
    ]

    weak_citations = [
        citation
        for citation in citations
        if citation.support_score < MINIMUM_CITATION_SUPPORT_SCORE
    ]

    failed_reasons: list[str] = []

    if invalid_citations:
        failed_reasons.append("citation_references_unretrieved_chunk")

    if weak_citations:
        failed_reasons.append("citation_support_score_below_threshold")

    valid_citation_count = len(citations) - len(invalid_citations)

    average_support_score = (
        sum(citation.support_score for citation in citations) / len(citations)
        if citations
        else 0.0
    )

    retrieval_validity_score = (
        valid_citation_count / len(citations)
        if citations
        else 0.0
    )

    citation_accuracy_score = round(
        average_support_score * retrieval_validity_score,
        4,
    )

    citation_check_passed = (
        not failed_reasons
        and citation_accuracy_score >= MINIMUM_CITATION_SUPPORT_SCORE
    )

    return CitationCheckResult(
        citations=citations,
        citation_check_passed=citation_check_passed,
        citation_accuracy_score=citation_accuracy_score,
        failed_reasons=failed_reasons,
        cited_chunk_ids=[
            citation.chunk_id
            for citation in citations
        ],
        valid_citation_count=valid_citation_count,
        total_citation_count=len(citations),
    )

def build_citations(
    answer: str,
    source_chunks: list[SourceChunk],
    max_citations: int,
)->list[Citation]:
    answer_terms = extract_keywords(answer)

    if not answer_terms:
        return []

    citations: list[Citation] = []

    for source_number, chunk in enumerate(source_chunks, start=1):
        chunk_terms = extract_keywords(chunk.text)
        matched_terms = sorted(answer_terms.intersection(chunk_terms))

        if not matched_terms:
            continue

        support_score = round(
            len(matched_terms) / len(answer_terms),
            4,
        )

        citations.append(
            Citation(
                source_number=source_number,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                retrieval_score=round(float(chunk.score), 4),
                support_score=support_score,
                matched_terms=matched_terms,
                text_excerpt=best_supporting_excerpt(
                    text=chunk.text,
                    answer_terms=answer_terms,
                ),
                metadata=chunk.metadata or {},
            )
        )

    citations.sort(
        key=lambda citation: (
            citation.support_score,
            citation.retrieval_score,
            -citation.source_number,
        ),
        reverse=True,
    )

    return citations[:max_citations]

def citation_to_dict(citation: Citation) -> dict[str, Any]:
    return {
        "source_number": citation.source_number,
        "chunk_id": citation.chunk_id,
        "document_id": citation.document_id,
        "retrieval_score": citation.retrieval_score,
        "support_score": citation.support_score,
        "matched_terms": citation.matched_terms,
        "text_excerpt": citation.text_excerpt,
        "metadata": citation.metadata,
    }

def citations_to_dicts(citations: list[Citation]) -> list[dict[str, Any]]:
    return [
        citation_to_dict(citation)
        for citation in citations
    ]

def best_supporting_excerpt(
    text: str,
    answer_terms: set[str],
    max_length: int = 280,
)->str:
    sentences = split_sentences(text)

    if not sentences:
        return truncate(normalize_text(text), max_length)

    ranked_sentences = []

    for sentence in sentences:
        sentence_terms = extract_keywords(sentence)
        matched_count = len(answer_terms.intersection(sentence_terms))

        ranked_sentences.append(
            (
                matched_count,
                len(sentence_terms),
                sentence,
            )
        )

    ranked_sentences.sort(reverse=True)

    best_sentence = ranked_sentences[0][2]

    return truncate(best_sentence, max_length)

def split_sentences(text: str) -> list[str]:
    cleaned_text = normalize_text(text)

    if not cleaned_text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", cleaned_text)

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]

def extract_keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    keywords: set[str] = set()

    for token in tokens:
        normalized_token = normalize_keyword(token)

        if not normalized_token:
            continue

        if normalized_token in STOP_WORDS:
            continue

        keywords.add(normalized_token)

    return keywords

def normalize_keyword(token: str) -> str:
    synonym_map = {
        "customers": "customer",
        "products": "product",
        "subscriptions": "subscription",
        "cancellations": "cancellation",
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

def is_fallback_answer(answer: str) -> bool:
    return normalize_text(answer).lower() == normalize_text(SAFE_FALLBACK_ANSWER).lower()

def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())

def truncate(text: str, max_length: int) -> str:
    cleaned = normalize_text(text)

    if len(cleaned) <= max_length:
        return cleaned

    return cleaned[: max_length - 3].rstrip() + "..."