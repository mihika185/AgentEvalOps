from backend.app.rag.answer_service import SimpleExtractiveAnswerGenerator
from backend.app.retrieval.retrieval_service import RetrievedChunk

def make_chunk(text: str, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk_test",
        document_id="doc_test",
        score=score,
        text=text,
        metadata={}
    )

def test_generator_extracts_relevant_answer_from_context():
    generator = SimpleExtractiveAnswerGenerator()

    answer = generator.generate_answer(
        query="When do subscription cancellations take effect?",
        chunks=[
            make_chunk(
                "Customers can request a refund within 7 days. "
                "Subscription cancellations take effect at the end of the billing cycle."
            )
        ]
    )

    assert "Subscription cancellations take effect" in answer
    assert "billing cycle" in answer

def test_generator_uses_retrieved_context_when_direct_match_is_weak():
    generator = SimpleExtractiveAnswerGenerator()

    answer = generator.generate_answer(
        query="Does the company offer space travel insurance?",
        chunks=[
            make_chunk(
                "Customers can request a refund within 7 days. "
                "Shipping charges are non-refundable."
            )
        ]
    )

    assert isinstance(answer, str)
    assert len(answer) > 0

def test_generator_returns_safe_fallback_when_no_chunks_are_available():
    generator = SimpleExtractiveAnswerGenerator()

    answer = generator.generate_answer(
        query="When do subscription cancellations take effect?",
        chunks=[]
    )

    assert "could not find enough reliable evidence" in answer.lower()