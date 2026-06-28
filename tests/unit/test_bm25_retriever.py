from backend.app.retrieval.bm25_retriever import (
    BM25CandidateChunk,
    score_chunks_with_bm25,
    tokenize_text,
)

def test_tokenize_text_normalizes_words_and_numbers():
    assert tokenize_text("Refunds: 14 calendar days!") == [
        "refunds",
        "14",
        "calendar",
        "days",
    ]

def test_bm25_scores_most_relevant_chunk_first():
    chunks = [
        BM25CandidateChunk(
            chunk_id="chunk_shipping",
            document_id="doc_1",
            chunk_index=0,
            text="Express shipping usually takes two business days for delivery.",
            metadata={},
        ),
        BM25CandidateChunk(
            chunk_id="chunk_refund",
            document_id="doc_1",
            chunk_index=1,
            text="Physical product refunds must be requested within 14 calendar days of delivery.",
            metadata={},
        ),
        BM25CandidateChunk(
            chunk_id="chunk_privacy",
            document_id="doc_1",
            chunk_index=2,
            text="Account deletion is completed after identity verification.",
            metadata={},
        ),
    ]
    scored_chunks = score_chunks_with_bm25(
        query="physical product refund delivery",
        chunks=chunks,
    )
    assert len(scored_chunks) >= 2
    assert scored_chunks[0].chunk_id == "chunk_refund"
    assert scored_chunks[0].score > scored_chunks[1].score