import re
from dataclasses import dataclass
from typing import Any


MINIMUM_CLAIM_SUPPORT_SCORE = 0.35

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
class ClaimSupportResult:
    claim: str
    support_score: float
    supported: bool
    best_source_id: str | None
    matched_terms: list[str]
    missing_terms: list[str]


@dataclass(frozen=True)
class FaithfulnessResult:
    faithfulness_score: float
    total_claim_count: int
    supported_claim_count: int
    unsupported_claim_count: int
    unsupported_claim_rate: float
    supported_claims: list[str]
    unsupported_claims: list[str]
    claim_results: list[ClaimSupportResult]


def evaluate_faithfulness(
    answer: str,
    source_chunks: list[Any],
    minimum_claim_support_score: float = MINIMUM_CLAIM_SUPPORT_SCORE,
) -> FaithfulnessResult:
    claims = extract_claims(answer)

    if not claims:
        return FaithfulnessResult(
            faithfulness_score=0.0,
            total_claim_count=0,
            supported_claim_count=0,
            unsupported_claim_count=0,
            unsupported_claim_rate=0.0,
            supported_claims=[],
            unsupported_claims=[],
            claim_results=[],
        )

    source_records = build_source_records(source_chunks)
    claim_results: list[ClaimSupportResult] = []

    for claim in claims:
        claim_results.append(
            score_claim_against_sources(
                claim=claim,
                source_records=source_records,
                minimum_claim_support_score=minimum_claim_support_score,
            )
        )

    supported_claims = [
        result.claim
        for result in claim_results
        if result.supported
    ]

    unsupported_claims = [
        result.claim
        for result in claim_results
        if not result.supported
    ]

    total_claim_count = len(claim_results)
    supported_claim_count = len(supported_claims)
    unsupported_claim_count = len(unsupported_claims)

    faithfulness_score = (
        supported_claim_count / total_claim_count
        if total_claim_count
        else 0.0
    )

    unsupported_claim_rate = (
        unsupported_claim_count / total_claim_count
        if total_claim_count
        else 0.0
    )

    return FaithfulnessResult(
        faithfulness_score=round(faithfulness_score, 4),
        total_claim_count=total_claim_count,
        supported_claim_count=supported_claim_count,
        unsupported_claim_count=unsupported_claim_count,
        unsupported_claim_rate=round(unsupported_claim_rate, 4),
        supported_claims=supported_claims,
        unsupported_claims=unsupported_claims,
        claim_results=claim_results,
    )


def score_claim_against_sources(
    claim: str,
    source_records: list[dict[str, Any]],
    minimum_claim_support_score: float,
) -> ClaimSupportResult:
    claim_terms = extract_keywords(claim)

    if not claim_terms:
        return ClaimSupportResult(
            claim=claim,
            support_score=0.0,
            supported=False,
            best_source_id=None,
            matched_terms=[],
            missing_terms=[],
        )

    best_score = 0.0
    best_source_id: str | None = None
    best_matched_terms: set[str] = set()

    for source in source_records:
        source_terms = source["terms"]
        matched_terms = claim_terms.intersection(source_terms)

        support_score = len(matched_terms) / len(claim_terms)

        if support_score > best_score:
            best_score = support_score
            best_source_id = source["source_id"]
            best_matched_terms = matched_terms

    missing_terms = claim_terms.difference(best_matched_terms)
    supported = best_score >= minimum_claim_support_score

    return ClaimSupportResult(
        claim=claim,
        support_score=round(best_score, 4),
        supported=supported,
        best_source_id=best_source_id,
        matched_terms=sorted(best_matched_terms),
        missing_terms=sorted(missing_terms),
    )


def build_source_records(source_chunks: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for chunk in source_chunks:
        source_id = str(
            getattr(chunk, "id", None)
            or getattr(chunk, "chunk_id", None)
            or "unknown_source"
        )

        text = str(
            getattr(chunk, "chunk_text", None)
            or getattr(chunk, "text", None)
            or ""
        )

        records.append(
            {
                "source_id": source_id,
                "text": text,
                "terms": extract_keywords(text),
            }
        )

    return records


def extract_claims(answer: str) -> list[str]:
    cleaned_answer = normalize_text(answer)

    if not cleaned_answer:
        return []

    parts = re.split(r"(?<=[.!?])\s+", cleaned_answer)
    claims: list[str] = []

    for part in parts:
        claim = normalize_claim(part)

        if not claim:
            continue

        if len(extract_keywords(claim)) < 2:
            continue

        claims.append(claim)

    return claims


def normalize_claim(claim: str) -> str:
    cleaned = normalize_text(claim)
    cleaned = re.sub(r"^[-*•]\s+", "", cleaned)
    cleaned = re.sub(r"^\d+[.)]\s+", "", cleaned)

    return cleaned


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


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())