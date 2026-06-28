from backend.app.retrieval.retrieval_metrics import (
    calculate_retrieval_metrics,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

def test_recall_at_k_counts_relevant_items_found():
    assert recall_at_k(
        retrieved_ids=["chunk_1", "chunk_2", "chunk_3"],
        relevant_ids=["chunk_2", "chunk_4"],
        k=3,
    ) == 0.5

def test_precision_at_k_uses_requested_k():
    assert precision_at_k(
        retrieved_ids=["chunk_1", "chunk_2", "chunk_3"],
        relevant_ids=["chunk_2", "chunk_3"],
        k=3,
    ) == 0.6667

def test_mrr_returns_first_relevant_rank():
    assert mean_reciprocal_rank(
        retrieved_ids=["chunk_1", "chunk_2", "chunk_3"],
        relevant_ids=["chunk_3"],
    ) == 0.3333

def test_ndcg_at_k_rewards_better_ranking():
    better_score = ndcg_at_k(
        retrieved_ids=["chunk_2", "chunk_1", "chunk_3"],
        relevant_ids=["chunk_2", "chunk_3"],
        k=3,
    )
    worse_score = ndcg_at_k(
        retrieved_ids=["chunk_1", "chunk_3", "chunk_2"],
        relevant_ids=["chunk_2", "chunk_3"],
        k=3,
    )
    assert better_score > worse_score

def test_calculate_retrieval_metrics_returns_scores():
    metrics = calculate_retrieval_metrics(
        retrieved_ids=["chunk_a", "chunk_b"],
        relevant_ids=["chunk_b"],
        k=2,
    )
    assert metrics == {
        "recall_at_2": 1.0,
        "precision_at_2": 0.5,
        "mrr": 0.5,
        "ndcg_at_2": 0.6309,
    }