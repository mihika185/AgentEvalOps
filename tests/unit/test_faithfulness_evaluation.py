from types import SimpleNamespace

from backend.app.evaluation.faithfulness_evaluator import (
    evaluate_faithfulness,
    extract_claims,
)
from backend.app.evaluation.hallucination_detector import detect_hallucination


def make_chunk(chunk_id: str, text: str):
    return SimpleNamespace(
        id=chunk_id,
        chunk_text=text,
    )

def test_extract_claims_splits_answer_into_claims():
    claims = extract_claims(
        "Customers must report damage within 48 hours. Refunds depend on inspection."
    )

    assert claims == [
        "Customers must report damage within 48 hours.",
        "Refunds depend on inspection.",
    ]

def test_faithfulness_passes_when_claims_are_supported():
    result = evaluate_faithfulness(
        answer=(
            "Customers must report damaged products within 48 hours. "
            "The report must include the order ID and clear photos."
        ),
        source_chunks=[
            make_chunk(
                "chunk_1",
                (
                    "Customers must report damaged products within 48 hours. "
                    "The report must include the order ID and clear photos."
                ),
            )
        ],
    )

    assert result.faithfulness_score == 1.0
    assert result.unsupported_claim_count == 0
    assert len(result.supported_claims) == 2

def test_faithfulness_detects_unsupported_claim():
    result = evaluate_faithfulness(
        answer=(
            "Customers must report damaged products within 48 hours. "
            "Customers receive a guaranteed refund within 30 days."
        ),
        source_chunks=[
            make_chunk(
                "chunk_1",
                "Customers must report damaged products within 48 hours.",
            )
        ],
    )

    assert result.faithfulness_score == 0.5
    assert result.unsupported_claim_count == 1
    assert "guaranteed refund" in result.unsupported_claims[0]


def test_hallucination_detector_stays_low_for_supported_answer():
    chunks = [
        make_chunk(
            "chunk_1",
            (
                "Customers must report damaged products within 48 hours. "
                "The report must include the order ID and clear photos."
            ),
        )
    ]

    faithfulness_result = evaluate_faithfulness(
        answer="Customers must report damaged products within 48 hours.",
        source_chunks=chunks,
    )

    result = detect_hallucination(
        answer="Customers must report damaged products within 48 hours.",
        source_chunks=chunks,
        faithfulness_result=faithfulness_result,
        citation_accuracy_score=1.0,
    )

    assert result.hallucination_detected is False
    assert result.hallucination_risk_score == 0.0
    assert result.risk_level == "low"


def test_hallucination_detector_flags_unsupported_claim():
    chunks = [
        make_chunk(
            "chunk_1",
            "Customers must report damaged products within 48 hours.",
        )
    ]

    answer = (
        "Customers must report damaged products within 48 hours. "
        "Customers receive a guaranteed refund within 30 days."
    )

    faithfulness_result = evaluate_faithfulness(
        answer=answer,
        source_chunks=chunks,
    )

    result = detect_hallucination(
        answer=answer,
        source_chunks=chunks,
        faithfulness_result=faithfulness_result,
        citation_accuracy_score=1.0,
    )

    assert result.hallucination_detected is True
    assert result.hallucination_risk_score >= 0.5
    assert result.risk_level == "high"
    assert "unsupported_claims_detected" in result.reasons

def test_supported_answer_is_not_hallucination_when_citation_score_is_moderate():
    chunks = [
        make_chunk(
            "chunk_1",
            (
                "ApexCart retries a failed renewal payment up to 3 times "
                "within 7 calendar days before suspending subscription benefits."
            ),
        )
    ]

    answer = (
        "ApexCart will retry a failed renewal payment up to 3 times within "
        "7 calendar days before suspending subscription benefits."
    )

    result = detect_hallucination(
        answer=answer,
        source_chunks=chunks,
        citation_accuracy_score=0.625,
    )

    assert result.hallucination_detected is False
    assert result.hallucination_risk_score == 0.0
    assert result.citation_penalty == 0.25
    assert "weak_or_missing_citations" not in result.reasons