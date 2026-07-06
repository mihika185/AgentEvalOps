import re
from dataclasses import dataclass
from typing import Any, Optional

from backend.app.evaluation.faithfulness_evaluator import (
    FaithfulnessResult,
    evaluate_faithfulness,
)


LOW_RISK_THRESHOLD = 0.20
HIGH_RISK_THRESHOLD = 0.50

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "should", "would", "will", "i", "you", "we", "they",
    "he", "she", "it", "this", "that", "these", "those", "to", "for",
    "of", "in", "on", "at", "by", "with", "and", "or", "but", "from",
    "as", "get", "be", "what", "when", "where", "who", "whom", "whose",
    "why", "how", "tell", "me", "about", "please", "policy", "policies",
    "company", "document", "documents", "answer", "answers", "question",
}


@dataclass(frozen=True)
class HallucinationResult:
    hallucination_detected: bool
    hallucination_risk_score: float
    risk_level: str
    unsupported_claim_rate: float
    unsupported_term_rate: float
    citation_penalty: float
    unsupported_claims: list[str]
    unsupported_terms: list[str]
    reasons: list[str]


def detect_hallucination(
    answer: str,
    source_chunks: list[Any],
    faithfulness_result: Optional[FaithfulnessResult] = None,
    citation_accuracy_score: Optional[float] = None,
) -> HallucinationResult:
    resolved_faithfulness = faithfulness_result or evaluate_faithfulness(
        answer=answer,
        source_chunks=source_chunks,
    )

    unsupported_term_rate, unsupported_terms = calculate_unsupported_term_rate(
        answer=answer,
        source_chunks=source_chunks,
    )

    citation_penalty = calculate_citation_penalty(citation_accuracy_score)

    risk_score = max(
        resolved_faithfulness.unsupported_claim_rate,
        unsupported_term_rate,
        citation_penalty,
    )

    risk_score = round(risk_score, 4)
    risk_level = classify_risk_level(risk_score)

    reasons: list[str] = []

    if resolved_faithfulness.unsupported_claim_count > 0:
        reasons.append("unsupported_claims_detected")

    if unsupported_term_rate >= LOW_RISK_THRESHOLD:
        reasons.append("answer_terms_not_supported_by_sources")

    if citation_penalty > 0:
        reasons.append("weak_or_missing_citations")

    hallucination_detected = risk_score >= LOW_RISK_THRESHOLD

    return HallucinationResult(
        hallucination_detected=hallucination_detected,
        hallucination_risk_score=risk_score,
        risk_level=risk_level,
        unsupported_claim_rate=resolved_faithfulness.unsupported_claim_rate,
        unsupported_term_rate=round(unsupported_term_rate, 4),
        citation_penalty=round(citation_penalty, 4),
        unsupported_claims=resolved_faithfulness.unsupported_claims,
        unsupported_terms=unsupported_terms,
        reasons=reasons,
    )


def calculate_unsupported_term_rate(
    answer: str,
    source_chunks: list[Any],
) -> tuple[float, list[str]]:
    answer_terms = extract_keywords(answer)
    source_terms = extract_source_terms(source_chunks)

    if not answer_terms:
        return 0.0, []

    unsupported_terms = sorted(answer_terms.difference(source_terms))

    unsupported_term_rate = len(unsupported_terms) / len(answer_terms)

    return unsupported_term_rate, unsupported_terms


def extract_source_terms(source_chunks: list[Any]) -> set[str]:
    source_terms: set[str] = set()

    for chunk in source_chunks:
        text = str(
            getattr(chunk, "chunk_text", None)
            or getattr(chunk, "text", None)
            or ""
        )

        source_terms.update(extract_keywords(text))

    return source_terms


def calculate_citation_penalty(citation_accuracy_score: Optional[float]) -> float:
    if citation_accuracy_score is None:
        return 0.0

    if citation_accuracy_score >= 0.80:
        return 0.0

    if citation_accuracy_score >= 0.50:
        return 0.25

    if citation_accuracy_score > 0.0:
        return 0.50

    return 1.0


def classify_risk_level(risk_score: float) -> str:
    if risk_score < LOW_RISK_THRESHOLD:
        return "low"

    if risk_score < HIGH_RISK_THRESHOLD:
        return "medium"

    return "high"


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
        "citations": "citation",
        "claims": "claim",
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