from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_POLICY_PATH = "sample_data/apexcart_customer_operations_policy.md"

DATASET_NAME = "ApexCart Policy Reliability Benchmark"

PIPELINE_TOPK3_NAME = "MiniLM Extractive top-k-3"
PIPELINE_TOPK5_NAME = "MiniLM Extractive top-k-5"


TEST_CASES = [
    {
        "question": "How long does a customer have to request a refund for a physical product?",
        "expected_behavior": "answerable",
        "expected_keywords": ["14 calendar days", "delivery"],
        "tags": ["refund", "direct"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "When does a monthly subscription cancellation take effect?",
        "expected_behavior": "answerable",
        "expected_keywords": ["end of the current billing cycle"],
        "tags": ["subscription", "direct"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "How many times will ApexCart retry a failed renewal payment?",
        "expected_behavior": "answerable",
        "expected_keywords": ["3 times", "7 calendar days"],
        "tags": ["subscription", "billing"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "What is the usual delivery time for express shipping?",
        "expected_behavior": "answerable",
        "expected_keywords": ["2 business days"],
        "tags": ["shipping", "direct"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "What must customers provide when reporting a damaged product?",
        "expected_behavior": "answerable",
        "expected_keywords": ["order id", "photos", "packaging"],
        "tags": ["damaged-items", "multi-fact"],
        "metadata_json": {"case_type": "multi_fact_answerable"},
    },
    {
        "question": "What does the hardware accessory warranty exclude?",
        "expected_behavior": "answerable",
        "expected_keywords": ["water damage", "unauthorized repairs"],
        "tags": ["warranty", "exclusion"],
        "metadata_json": {"case_type": "multi_fact_answerable"},
    },
    {
        "question": "How long does account deletion take after identity verification?",
        "expected_behavior": "answerable",
        "expected_keywords": ["30 calendar days"],
        "tags": ["account-deletion", "privacy"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "For how long may ApexCart retain tax and fraud review records?",
        "expected_behavior": "answerable",
        "expected_keywords": ["7 years"],
        "tags": ["retention", "privacy"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "What happens if a customer files a chargeback before contacting support?",
        "expected_behavior": "answerable",
        "expected_keywords": ["temporarily paused"],
        "tags": ["billing", "chargeback"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "When can a customer request escalation for an unresolved support issue?",
        "expected_behavior": "answerable",
        "expected_keywords": ["3 business days"],
        "tags": ["support", "escalation"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "Does ApexCart accept cryptocurrency payments for subscriptions?",
        "expected_behavior": "unanswerable",
        "expected_keywords": [],
        "tags": ["unanswerable", "payments"],
        "metadata_json": {"case_type": "unanswerable"},
    },
    {
        "question": "Does ApexCart offer a student discount for annual plans?",
        "expected_behavior": "unanswerable",
        "expected_keywords": [],
        "tags": ["unanswerable", "discounts"],
        "metadata_json": {"case_type": "unanswerable"},
    },
]


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    **kwargs: Any,
) -> Any:
    response = client.request(method, url, **kwargs)

    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = response.text

    if response.status_code >= 400:
        raise RuntimeError(
            f"{method} {url} failed with status {response.status_code}:\n{payload}"
        )

    return payload


def upload_document(
    client: httpx.Client,
    base_url: str,
    policy_path: Path,
) -> str:
    with policy_path.open("rb") as file_obj:
        payload = request_json(
            client,
            "POST",
            f"{base_url}/api/v1/documents/upload-and-index",
            files={"file": (policy_path.name, file_obj, "text/markdown")},
        )

    document_id = payload.get("id")

    if not document_id:
        raise RuntimeError(f"Document upload response did not contain id: {payload}")

    return str(document_id)


def create_dataset(
    client: httpx.Client,
    base_url: str,
    document_id: str,
) -> str:
    payload = request_json(
        client,
        "POST",
        f"{base_url}/api/v1/benchmarks/datasets",
        json={
            "name": DATASET_NAME,
            "description": (
                "Synthetic enterprise policy benchmark for testing RAG reliability, "
                "quality gates, and failure analysis."
            ),
            "document_id": document_id,
            "metadata_json": {
                "corpus_type": "synthetic_enterprise_policy",
                "purpose": "rag_reliability_evaluation",
                "seeded_by": "scripts/seed_apexcart_benchmark.py",
            },
        },
    )

    dataset_id = payload.get("id")

    if not dataset_id:
        raise RuntimeError(f"Dataset creation response did not contain id: {payload}")

    return str(dataset_id)


def add_test_cases(
    client: httpx.Client,
    base_url: str,
    dataset_id: str,
) -> None:
    for test_case in TEST_CASES:
        request_json(
            client,
            "POST",
            f"{base_url}/api/v1/benchmarks/datasets/{dataset_id}/test-cases",
            json=test_case,
        )


def list_pipeline_configs(
    client: httpx.Client,
    base_url: str,
) -> list[dict[str, Any]]:
    payload = request_json(
        client,
        "GET",
        f"{base_url}/api/v1/pipeline-configs",
    )

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("items", "pipeline_configs", "configs", "data"):
            value = payload.get(key)

            if isinstance(value, list):
                return value

    return []


def find_pipeline_config_id(
    configs: list[dict[str, Any]],
    name: str,
) -> str | None:
    for config in configs:
        if config.get("name") == name and config.get("id"):
            return str(config["id"])

    return None


def create_pipeline_config(
    client: httpx.Client,
    base_url: str,
    name: str,
    top_k: int,
) -> str:
    payload = request_json(
        client,
        "POST",
        f"{base_url}/api/v1/pipeline-configs",
        json={
            "name": name,
            "description": (
                f"Default extractive RAG pipeline using MiniLM embeddings "
                f"and top-k {top_k} retrieval."
            ),
            "retrieval_provider": "qdrant_vector_search",
            "top_k": top_k,
            "answer_generator_provider": "extractive",
            "answer_generator_model": "simple-extractive-v2",
            "embedding_provider": "sentence-transformers",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "quality_gate_profile": "default-v1",
            "is_active": True,
            "metadata_json": {
                "purpose": "apexcart_benchmark_comparison",
                "seeded_by": "scripts/seed_apexcart_benchmark.py",
            },
        },
    )

    config_id = payload.get("id")

    if not config_id:
        raise RuntimeError(f"Pipeline config creation response did not contain id: {payload}")

    return str(config_id)


def ensure_pipeline_config(
    client: httpx.Client,
    base_url: str,
    name: str,
    top_k: int,
) -> str:
    configs = list_pipeline_configs(client, base_url)
    existing_id = find_pipeline_config_id(configs, name)

    if existing_id:
        return existing_id

    return create_pipeline_config(
        client=client,
        base_url=base_url,
        name=name,
        top_k=top_k,
    )


def run_comparison(
    client: httpx.Client,
    base_url: str,
    dataset_id: str,
    pipeline_config_ids: list[str],
) -> dict[str, Any]:
    payload = request_json(
        client,
        "POST",
        f"{base_url}/api/v1/benchmarks/datasets/{dataset_id}/compare",
        json={"pipeline_config_ids": pipeline_config_ids},
    )

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected comparison response: {payload}")

    return payload


def print_summary(
    document_id: str,
    dataset_id: str,
    topk3_id: str,
    topk5_id: str,
    comparison: dict[str, Any],
) -> None:
    print("\nSeed completed successfully.")
    print(f"Document ID: {document_id}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Top-k-3 pipeline ID: {topk3_id}")
    print(f"Top-k-5 pipeline ID: {topk5_id}")

    print("\nComparison result:")
    print(f"Best pipeline: {comparison.get('best_pipeline_config_name')}")
    print(f"Best pipeline ID: {comparison.get('best_pipeline_config_id')}")
    print(f"Selection reason: {comparison.get('selection_reason')}")

    print("\nScores:")

    for result in comparison.get("results", []):
        run = result.get("benchmark_run", {})

        print(
            "- {name}: {passed}/{total} passed, "
            "answerable={answerable_passed}/{answerable_total}, "
            "unanswerable={unanswerable_passed}/{unanswerable_total}, "
            "pass_rate={pass_rate}".format(
                name=result.get("pipeline_config_name"),
                passed=run.get("passed_cases"),
                total=run.get("total_cases"),
                answerable_passed=run.get("answerable_passed"),
                answerable_total=run.get("answerable_cases"),
                unanswerable_passed=run.get("unanswerable_passed"),
                unanswerable_total=run.get("unanswerable_cases"),
                pass_rate=run.get("pass_rate"),
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed and run the ApexCart RAG reliability benchmark."
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL. Default: {DEFAULT_BASE_URL}",
    )

    parser.add_argument(
        "--policy-path",
        default=DEFAULT_POLICY_PATH,
        help=f"Path to the sample policy file. Default: {DEFAULT_POLICY_PATH}",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_url = args.base_url.rstrip("/")
    policy_path = Path(args.policy_path)

    if not policy_path.exists():
        raise FileNotFoundError(
            f"Policy file not found: {policy_path}. "
            "Create it or pass --policy-path."
        )

    with httpx.Client(timeout=60.0) as client:
        print("Uploading and indexing policy document...")
        document_id = upload_document(client, base_url, policy_path)

        print("Creating benchmark dataset...")
        dataset_id = create_dataset(client, base_url, document_id)

        print("Adding benchmark test cases...")
        add_test_cases(client, base_url, dataset_id)

        print("Ensuring pipeline configs...")
        topk3_id = ensure_pipeline_config(
            client=client,
            base_url=base_url,
            name=PIPELINE_TOPK3_NAME,
            top_k=3,
        )

        topk5_id = ensure_pipeline_config(
            client=client,
            base_url=base_url,
            name=PIPELINE_TOPK5_NAME,
            top_k=5,
        )

        print("Running benchmark comparison...")
        comparison = run_comparison(
            client=client,
            base_url=base_url,
            dataset_id=dataset_id,
            pipeline_config_ids=[topk3_id, topk5_id],
        )

    print_summary(
        document_id=document_id,
        dataset_id=dataset_id,
        topk3_id=topk3_id,
        topk5_id=topk5_id,
        comparison=comparison,
    )


if __name__ == "__main__":
    main()
