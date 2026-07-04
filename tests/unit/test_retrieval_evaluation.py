import pytest

from backend.app.evaluation.retrieval_metrics import (
    calculate_retrieval_metrics,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k():
    assert recall_at_k(
        retrieved_ids=["chunk_1", "chunk_2", "chunk_3"],
        relevant_ids=["chunk_2", "chunk_4"],
        k=3,
    ) == 0.5


def test_precision_at_k():
    assert precision_at_k(
        retrieved_ids=["chunk_1", "chunk_2", "chunk_3"],
        relevant_ids=["chunk_2", "chunk_4"],
        k=3,
    ) == 0.3333


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank(
        retrieved_ids=["chunk_1", "chunk_2", "chunk_3"],
        relevant_ids=["chunk_2"],
    ) == 0.5


def test_ndcg_at_k_perfect_first_match():
    assert ndcg_at_k(
        retrieved_ids=["chunk_2", "chunk_1", "chunk_3"],
        relevant_ids=["chunk_2"],
        k=3,
    ) == 1.0


def test_calculate_retrieval_metrics():
    metrics = calculate_retrieval_metrics(
        retrieved_ids=["chunk_1", "chunk_2", "chunk_3"],
        relevant_ids=["chunk_2"],
        k=3,
    )

    assert metrics == {
        "recall_at_3": 1.0,
        "precision_at_3": 0.3333,
        "mrr": 0.5,
        "ndcg_at_3": 0.6309,
    }


def test_metrics_reject_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k(
            retrieved_ids=["chunk_1"],
            relevant_ids=["chunk_1"],
            k=0,
        )