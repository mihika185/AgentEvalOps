from backend.app.rag.answer_service import SAFE_FALLBACK_ANSWER, SourceChunk
from backend.app.rag.citation_checker import (
    best_supporting_excerpt,
    build_citations,
    check_answer_citations,
    extract_keywords,
)


def make_source_chunk(
    chunk_id: str,
    text: str,
    score: float = 0.91,
) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        document_id="doc_test",
        score=score,
        text=text,
        metadata={
            "page": 1,
        },
    )


def test_extract_keywords_normalizes_basic_terms():
    keywords = extract_keywords(
        "Customers reported damaged products within 7 days."
    )

    assert "report" in keywords
    assert "damage" in keywords
    assert "product" in keywords
    assert "day" in keywords
    assert "within" in keywords
    assert "customer" not in keywords


def test_build_citations_selects_supporting_chunk():
    answer = "Damaged products must be reported within 7 days."
    chunks = [
        make_source_chunk(
            chunk_id="chunk_1",
            text="Shipping speed depends on the destination.",
            score=0.8,
        ),
        make_source_chunk(
            chunk_id="chunk_2",
            text="Customers must report damaged products within 7 days of delivery.",
            score=0.7,
        ),
    ]

    citations = build_citations(
        answer=answer,
        source_chunks=chunks,
        max_citations=2,
    )

    assert citations
    assert citations[0].chunk_id == "chunk_2"
    assert citations[0].support_score > 0
    assert "damage" in citations[0].matched_terms


def test_check_answer_citations_passes_for_supported_answer():
    answer = "Customers must report damaged products within 7 days of delivery."
    chunks = [
        make_source_chunk(
            chunk_id="chunk_1",
            text="Customers must report damaged products within 7 days of delivery.",
        )
    ]

    result = check_answer_citations(
        answer=answer,
        source_chunks=chunks,
    )

    assert result.citation_check_passed is True
    assert result.citation_accuracy_score > 0
    assert result.valid_citation_count == 1
    assert result.total_citation_count == 1
    assert result.cited_chunk_ids == ["chunk_1"]


def test_check_answer_citations_fails_for_fallback_answer():
    result = check_answer_citations(
        answer=SAFE_FALLBACK_ANSWER,
        source_chunks=[
            make_source_chunk(
                chunk_id="chunk_1",
                text="Customers must report damaged products within 7 days.",
            )
        ],
    )

    assert result.citation_check_passed is False
    assert result.citation_accuracy_score == 0.0
    assert result.failed_reasons == ["answer_is_safe_fallback"]


def test_best_supporting_excerpt_returns_relevant_sentence():
    excerpt = best_supporting_excerpt(
        text=(
            "Shipping speed depends on destination. "
            "Customers must report damaged products within 7 days."
        ),
        answer_terms={"damage", "product", "day", "report"},
    )

    assert "damaged products" in excerpt