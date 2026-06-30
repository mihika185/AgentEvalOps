from dataclasses import dataclass

from backend.app.retrieval.reranker import (
    DEFAULT_CROSS_ENCODER_MODEL,
    rerank_chunks,
    tokenize_text,
)

@dataclass(frozen=True)
class FakeChunk:
    chunk_id: str
    text: str
    score: float

def test_tokenize_text_normalizes_words_and_numbers():
    assert tokenize_text("Order ID: 48-hours!") == [
        "order",
        "id",
        "48",
        "hours",
    ]

def test_cross_encoder_reranker_uses_model_scores_without_downloading_model():
    chunks = [
        FakeChunk(
            chunk_id="shipping",
            text="Express shipping usually takes two business days.",
            score=0.95,
        ),
        FakeChunk(
            chunk_id="damaged_product",
            text="Customers must include order ID, photos, packaging, and issue description for damaged products.",
            score=0.2,
        ),
    ]

    def fake_scorer(pairs, model_name):
        assert model_name == DEFAULT_CROSS_ENCODER_MODEL
        assert len(pairs) == 2

        return [0.1, 8.0]

    ranked = rerank_chunks(
        query="damaged product order id photos packaging",
        chunks=chunks,
        cross_encoder_scorer=fake_scorer,
    )

    assert ranked[0].chunk.chunk_id == "damaged_product"
    assert ranked[0].score == 1.0
    assert ranked[0].details["reranker_provider"] == "cross_encoder"
    assert ranked[0].details["reranker_model"] == DEFAULT_CROSS_ENCODER_MODEL
    assert ranked[0].details["raw_cross_encoder_score"] == 8.0

def test_term_overlap_fallback_moves_stronger_text_match_first():
    chunks = [
        FakeChunk(
            chunk_id="shipping",
            text="Express shipping usually takes two business days.",
            score=0.95,
        ),
        FakeChunk(
            chunk_id="damaged_product",
            text="Customers must include order ID, photos, packaging, and issue description for damaged products.",
            score=0.2,
        ),
    ]

    ranked = rerank_chunks(
        query="damaged product order id photos packaging",
        chunks=chunks,
        provider="term_overlap",
    )

    assert ranked[0].chunk.chunk_id == "damaged_product"
    assert ranked[0].score > ranked[1].score
    assert ranked[0].details["reranker_provider"] == "term_overlap"
    assert "photos" in ranked[0].details["matched_terms"]

def test_rerank_chunks_falls_back_when_cross_encoder_fails():
    chunks = [
        FakeChunk(
            chunk_id="shipping",
            text="Express shipping usually takes two business days.",
            score=0.95,
        ),
        FakeChunk(
            chunk_id="damaged_product",
            text="Customers must include order ID, photos, packaging, and issue description for damaged products.",
            score=0.2,
        ),
    ]

    def broken_scorer(pairs, model_name):
        raise RuntimeError("model unavailable")

    ranked = rerank_chunks(
        query="damaged product order id photos packaging",
        chunks=chunks,
        cross_encoder_scorer=broken_scorer,
    )

    assert ranked[0].chunk.chunk_id == "damaged_product"
    assert ranked[0].details["reranker_provider"] == "term_overlap"
    assert ranked[0].details["fallback_from"] == "cross_encoder_v1"
    assert "model unavailable" in ranked[0].details["fallback_reason"]

def test_rerank_chunks_returns_empty_list_for_no_chunks():
    assert rerank_chunks("refund policy", []) == []

def test_cross_encoder_reranker_filters_low_raw_scores():
    chunks = [
        FakeChunk(
            chunk_id="weak_negative",
            text="Billing disputes are reviewed within 30 calendar days.",
            score=0.9,
        ),
        FakeChunk(
            chunk_id="strong_positive",
            text="Customers must include order ID, photos, packaging, and issue description for damaged products.",
            score=0.2,
        ),
        FakeChunk(
            chunk_id="weak_borderline",
            text="Express shipping usually takes two business days.",
            score=0.7,
        ),
    ]

    def fake_scorer(pairs, model_name):
        return [-2.0, 7.0, -0.1]

    ranked = rerank_chunks(
        query="What must customers provide when reporting a damaged product?",
        chunks=chunks,
        cross_encoder_scorer=fake_scorer,
    )

    assert len(ranked) == 1
    assert ranked[0].chunk.chunk_id == "strong_positive"
    assert ranked[0].details["raw_cross_encoder_score"] == 7.0
    assert ranked[0].details["min_cross_encoder_score"] == 0.0
    assert ranked[0].details["filtered_by_cross_encoder_score"] is True

def test_cross_encoder_reranker_keeps_best_chunk_if_all_scores_are_below_threshold():
    chunks = [
        FakeChunk(
            chunk_id="least_bad",
            text="Customers may contact support for order issues.",
            score=0.4,
        ),
        FakeChunk(
            chunk_id="worse",
            text="Express shipping usually takes two business days.",
            score=0.9,
        ),
    ]

    def fake_scorer(pairs, model_name):
        return [-0.2, -3.0]

    ranked = rerank_chunks(
        query="What must customers provide when reporting a damaged product?",
        chunks=chunks,
        cross_encoder_scorer=fake_scorer,
    )

    assert len(ranked) == 1
    assert ranked[0].chunk.chunk_id == "least_bad"
    assert ranked[0].details["kept_as_best_available_result"] is True
    assert ranked[0].details["filtered_by_cross_encoder_score"] is True