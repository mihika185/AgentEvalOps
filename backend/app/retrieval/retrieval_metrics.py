from __future__ import annotations
import math

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    retrieved_at_k = set(retrieved_ids[:k])
    return round(len(retrieved_at_k & relevant) / len(relevant), 4)

def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if not retrieved_ids:
        return 0.0
    relevant = set(relevant_ids)
    retrieved_at_k = retrieved_ids[:k]
    if not retrieved_at_k:
        return 0.0
    hits = sum(item_id in relevant for item_id in retrieved_at_k)
    return round(hits / k, 4)

def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    for index, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant:
            return round(1.0 / index, 4)
    return 0.0

def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = 0.0
    for index, item_id in enumerate(retrieved_ids[:k], start=1):
        if item_id in relevant:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    if ideal_dcg == 0:
        return 0.0
    return round(dcg / ideal_dcg, 4)

def calculate_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> dict[str, float]:
    return {
        f"recall_at_{k}": recall_at_k(retrieved_ids, relevant_ids, k),
        f"precision_at_{k}": precision_at_k(retrieved_ids, relevant_ids, k),
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
        f"ndcg_at_{k}": ndcg_at_k(retrieved_ids, relevant_ids, k),
    }