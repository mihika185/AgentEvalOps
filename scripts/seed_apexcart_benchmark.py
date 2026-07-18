from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_POLICY_PATH = "sample_data/apexcart_customer_operations_policy.md"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_COMPARISON_TIMEOUT_SECONDS = 900.0

DATASET_NAME = "ApexCart Policy Reliability Benchmark"
SEED_SCRIPT_NAME = "scripts/seed_apexcart_benchmark.py"

DEFAULT_PIPELINE_NAMES = [
    "BM25 Baseline",
    "Dense Retrieval",
    "Hybrid Retrieval",
    "Hybrid Retrieval + Rerank",
]

TEST_CASES = [
    {
        "question": "How long does a customer have to request a refund for a physical product?",
        "expected_behavior": "answerable",
        "expected_keywords": ["14 calendar days", "delivery"],
        "relevant_text_markers": [
            "Customers may request a refund for physical products within 14 calendar days of delivery."
        ],
        "tags": ["refund", "direct"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "When does a monthly subscription cancellation take effect?",
        "expected_behavior": "answerable",
        "expected_keywords": ["end of the current billing cycle"],
        "relevant_text_markers": [
            "Monthly subscription cancellations take effect at the end of the current billing cycle."
        ],
        "tags": ["subscription", "direct"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "How many times will ApexCart retry a failed renewal payment?",
        "expected_behavior": "answerable",
        "expected_keywords": ["3 times", "7 calendar days"],
        "relevant_text_markers": [
            "ApexCart will retry the payment up to 3 times within 7 calendar days"
        ],
        "tags": ["subscription", "billing"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "What is the usual delivery time for express shipping?",
        "expected_behavior": "answerable",
        "expected_keywords": ["2 business days"],
        "relevant_text_markers": [
            "Express shipping usually takes 2 business days after order dispatch."
        ],
        "tags": ["shipping", "direct"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "What must customers provide when reporting a damaged product?",
        "expected_behavior": "answerable",
        "expected_keywords": ["order id", "photos", "packaging"],
        "relevant_text_markers": [
            "The report must include the order ID, clear photos of the damaged product, photos of the outer packaging"
        ],
        "tags": ["damaged-items", "multi-fact"],
        "metadata_json": {"case_type": "multi_fact_answerable"},
    },
    {
        "question": "What does the hardware accessory warranty exclude?",
        "expected_behavior": "answerable",
        "expected_keywords": ["water damage", "unauthorized repairs"],
        "relevant_text_markers": [
            "The warranty does not cover water damage",
            "unauthorized repairs",
        ],
        "tags": ["warranty", "exclusion"],
        "metadata_json": {"case_type": "multi_fact_answerable"},
    },
    {
        "question": "How long does account deletion take after identity verification?",
        "expected_behavior": "answerable",
        "expected_keywords": ["30 calendar days"],
        "relevant_text_markers": [
            "Account deletion is processed within 30 calendar days after identity verification is completed."
        ],
        "tags": ["account-deletion", "privacy"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "For how long may ApexCart retain tax and fraud review records?",
        "expected_behavior": "answerable",
        "expected_keywords": ["7 years"],
        "relevant_text_markers": [
            "ApexCart may retain invoices, tax records, fraud review logs, chargeback evidence, and compliance records for up to 7 years"
        ],
        "tags": ["retention", "privacy"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "What happens if a customer files a chargeback before contacting support?",
        "expected_behavior": "answerable",
        "expected_keywords": ["temporarily paused"],
        "relevant_text_markers": [
            "If a customer files a chargeback before contacting ApexCart support, the related account may be temporarily paused"
        ],
        "tags": ["billing", "chargeback"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "When can a customer request escalation for an unresolved support issue?",
        "expected_behavior": "answerable",
        "expected_keywords": ["3 business days"],
        "relevant_text_markers": [
            "Customers may request escalation if an issue remains unresolved after 3 business days from the first support response."
        ],
        "tags": ["support", "escalation"],
        "metadata_json": {"case_type": "direct_answerable"},
    },
    {
        "question": "Does ApexCart accept cryptocurrency payments for subscriptions?",
        "expected_behavior": "unanswerable",
        "expected_keywords": [],
        "relevant_text_markers": [],
        "tags": ["unanswerable", "payments"],
        "metadata_json": {"case_type": "unanswerable"},
    },
    {
        "question": "Does ApexCart offer a student discount for annual plans?",
        "expected_behavior": "unanswerable",
        "expected_keywords": [],
        "relevant_text_markers": [],
        "tags": ["unanswerable", "discounts"],
        "metadata_json": {"case_type": "unanswerable"},
    },
]

def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    request_timeout: float | httpx.Timeout | None = None,
    **kwargs: Any,
) -> Any:
    request_kwargs = dict(kwargs)

    if request_timeout is not None:
        request_kwargs["timeout"] = request_timeout

    try:
        response = client.request(
            method,
            url,
            **request_kwargs,
        )
    except httpx.ReadTimeout as exc:
        timeout_value = (
            request_timeout
            if request_timeout is not None
            else client.timeout.read
        )

        raise RuntimeError(
            f"{method} {url} exceeded the configured read timeout "
            f"of {timeout_value} seconds. The API may still be processing "
            "the request, so inspect recent benchmark runs before retrying."
        ) from exc

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


def list_datasets(
    client: httpx.Client,
    base_url: str,
) -> list[dict[str, Any]]:
    payload = request_json(
        client,
        "GET",
        f"{base_url}/api/v1/benchmarks/datasets",
    )

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("items", "datasets", "data"):
            value = payload.get(key)

            if isinstance(value, list):
                return value

    return []


def find_seeded_dataset(
    datasets: list[dict[str, Any]],
) -> tuple[str, str] | None:
    matching_datasets: list[dict[str, Any]] = []

    for dataset in datasets:
        metadata = dataset.get("metadata_json") or {}

        if dataset.get("name") != DATASET_NAME:
            continue

        if metadata.get("seeded_by") != SEED_SCRIPT_NAME:
            continue

        if not dataset.get("id") or not dataset.get("document_id"):
            continue

        matching_datasets.append(dataset)

    if not matching_datasets:
        return None

    matching_datasets.sort(
        key=lambda item: str(item.get("created_at", "")),
        reverse=True,
    )

    latest_dataset = matching_datasets[0]

    return str(latest_dataset["id"]), str(latest_dataset["document_id"])


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
                "quality gates, provider-backed answer generation, and failure analysis."
            ),
            "document_id": document_id,
            "metadata_json": {
                "corpus_type": "synthetic_enterprise_policy",
                "purpose": "rag_reliability_evaluation",
                "seeded_by": SEED_SCRIPT_NAME,
            },
        },
    )

    dataset_id = payload.get("id")

    if not dataset_id:
        raise RuntimeError(f"Dataset creation response did not contain id: {payload}")

    return str(dataset_id)


def get_or_create_dataset(
    client: httpx.Client,
    base_url: str,
    policy_path: Path,
) -> tuple[str, str]:
    datasets = list_datasets(client, base_url)
    existing_dataset = find_seeded_dataset(datasets)

    if existing_dataset:
        dataset_id, document_id = existing_dataset

        print(f"Reusing existing benchmark dataset: {dataset_id}")
        print(f"Reusing existing indexed document: {document_id}")

        return document_id, dataset_id

    print("No seeded benchmark dataset found.")
    print("Uploading and indexing policy document...")

    document_id = upload_document(
        client=client,
        base_url=base_url,
        policy_path=policy_path,
    )

    print("Creating benchmark dataset...")

    dataset_id = create_dataset(
        client=client,
        base_url=base_url,
        document_id=document_id,
    )

    return document_id, dataset_id

def list_document_chunks(
    client: httpx.Client,
    base_url: str,
    document_id: str,
) -> list[dict[str, Any]]:
    payload = request_json(
        client,
        "GET",
        f"{base_url}/api/v1/documents/{document_id}/chunks",
        params={"limit": 200},
    )

    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected document chunks response: {payload}")

    return payload

def normalize_evidence_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())

def resolve_relevant_chunk_ids(
    test_case: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> list[str]:
    markers = test_case.get("relevant_text_markers") or []

    if not markers:
        return []

    normalized_chunks = [
        (
            str(chunk.get("id") or "").strip(),
            normalize_evidence_text(chunk.get("chunk_text", "")),
        )
        for chunk in chunks
    ]

    relevant_chunk_ids: list[str] = []

    for marker in markers:
        normalized_marker = normalize_evidence_text(marker)
        matching_chunk_ids = [
            chunk_id
            for chunk_id, chunk_text in normalized_chunks
            if chunk_id and normalized_marker in chunk_text
        ]

        if not matching_chunk_ids:
            raise RuntimeError(
                "Could not resolve retrieval ground truth for question "
                f"'{test_case['question']}'. Missing marker: {marker}"
            )

        for chunk_id in matching_chunk_ids:
            if chunk_id not in relevant_chunk_ids:
                relevant_chunk_ids.append(chunk_id)

    return relevant_chunk_ids

def build_test_case_payload(
    test_case: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    markers = list(test_case.get("relevant_text_markers") or [])
    relevant_chunk_ids = resolve_relevant_chunk_ids(test_case, chunks)
    metadata = {
        **(test_case.get("metadata_json") or {}),
        "expected_relevant_chunk_ids": relevant_chunk_ids,
        "ground_truth_method": "exact_text_marker_match_v1",
        "ground_truth_markers": markers,
    }

    return {
        "question": test_case["question"],
        "expected_behavior": test_case["expected_behavior"],
        "expected_keywords": test_case["expected_keywords"],
        "tags": test_case["tags"],
        "metadata_json": metadata,
    }

def list_test_cases(
    client: httpx.Client,
    base_url: str,
    dataset_id: str,
) -> list[dict[str, Any]]:
    payload = request_json(
        client,
        "GET",
        f"{base_url}/api/v1/benchmarks/datasets/{dataset_id}/test-cases",
    )

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("items", "test_cases", "cases", "data"):
            value = payload.get(key)

            if isinstance(value, list):
                return value

    return []

def add_test_cases(
    client: httpx.Client,
    base_url: str,
    dataset_id: str,
    document_id: str,
) -> None:
    chunks = list_document_chunks(
        client=client,
        base_url=base_url,
        document_id=document_id,
    )

    if not chunks:
        raise RuntimeError(
            f"Indexed document '{document_id}' does not contain any chunks"
        )

    existing_cases = list_test_cases(
        client=client,
        base_url=base_url,
        dataset_id=dataset_id,
    )

    existing_cases_by_question = {
        str(test_case.get("question")): test_case
        for test_case in existing_cases
        if test_case.get("question")
    }

    created_count = 0
    updated_count = 0

    for test_case in TEST_CASES:
        question = test_case["question"]
        payload = build_test_case_payload(test_case, chunks)
        existing_case = existing_cases_by_question.get(question)

        if existing_case is not None:
            test_case_id = existing_case.get("id")

            if not test_case_id:
                raise RuntimeError(
                    f"Existing benchmark test case is missing id: {existing_case}"
                )

            request_json(
                client,
                "PATCH",
                f"{base_url}/api/v1/benchmarks/test-cases/{test_case_id}",
                json=payload,
            )

            updated_count += 1
            continue

        request_json(
            client,
            "POST",
            f"{base_url}/api/v1/benchmarks/datasets/{dataset_id}/test-cases",
            json=payload,
        )

        created_count += 1

    print(
        f"Benchmark test cases: {created_count} created, "
        f"{updated_count} updated."
    )

def ensure_default_pipeline_configs(
    client: httpx.Client,
    base_url: str,
) -> dict[str, str]:
    payload = request_json(
        client,
        "POST",
        f"{base_url}/api/v1/pipeline-configs/defaults",
    )

    configs = payload.get("configs")

    if not isinstance(configs, list):
        raise RuntimeError(f"Unexpected default pipeline response: {payload}")

    config_ids: dict[str, str] = {}

    for config in configs:
        name = config.get("name")
        config_id = config.get("id")

        if name in DEFAULT_PIPELINE_NAMES and config_id:
            config_ids[str(name)] = str(config_id)

    missing_names = [
        name
        for name in DEFAULT_PIPELINE_NAMES
        if name not in config_ids
    ]

    if missing_names:
        raise RuntimeError(
            "Default pipeline configs missing from API response: "
            + ", ".join(missing_names)
        )

    print(
        "Default pipeline configs ready: "
        f"{payload.get('created_count', 0)} created, "
        f"{payload.get('updated_count', 0)} updated."
    )

    return config_ids

def run_comparison(
    client: httpx.Client,
    base_url: str,
    dataset_id: str,
    pipeline_config_ids: list[str],
    comparison_timeout_seconds: float,
) -> dict[str, Any]:
    payload = request_json(
        client,
        "POST",
        f"{base_url}/api/v1/benchmarks/datasets/{dataset_id}/compare",
        request_timeout=comparison_timeout_seconds,
        json={"pipeline_config_ids": pipeline_config_ids},
    )

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected comparison response: {payload}")

    return payload

def print_summary(
    document_id: str,
    dataset_id: str,
    pipeline_ids: dict[str, str],
    comparison: dict[str, Any],
) -> None:
    print("\nSeed completed successfully.")
    print(f"Document ID: {document_id}")
    print(f"Dataset ID: {dataset_id}")

    print("\nPipeline IDs:")

    for name, config_id in pipeline_ids.items():
        print(f"- {name}: {config_id}")

    print("\nComparison result:")
    print(f"Best pipeline: {comparison.get('best_pipeline_config_name')}")
    print(f"Best pipeline ID: {comparison.get('best_pipeline_config_id')}")
    print(f"Selection reason: {comparison.get('selection_reason')}")

    print("\nScores:")

    for result in comparison.get("results", []):
        run = result.get("benchmark_run", {})
        run_metadata = run.get("metadata_json") or {}

        print(
            "- {name} [{retrieval}, {generator}:{model}]: {passed}/{total} passed, "
            "answerable={answerable_passed}/{answerable_total}, "
            "unanswerable={unanswerable_passed}/{unanswerable_total}, "
            "pass_rate={pass_rate}, avg_quality={avg_quality}, "
            "avg_latency_ms={avg_latency}, avg_cost={avg_cost}".format(
                name=result.get("pipeline_config_name"),
                retrieval=run_metadata.get("retrieval_provider"),
                generator=run_metadata.get("answer_generator_provider"),
                model=run_metadata.get("answer_generator_model"),
                passed=run.get("passed_cases"),
                total=run.get("total_cases"),
                answerable_passed=run.get("answerable_passed"),
                answerable_total=run.get("answerable_cases"),
                unanswerable_passed=run.get("unanswerable_passed"),
                unanswerable_total=run.get("unanswerable_cases"),
                pass_rate=run.get("pass_rate"),
                avg_quality=run.get("average_overall_quality_score"),
                avg_latency=run.get("average_latency_ms"),
                avg_cost=run_metadata.get("average_estimated_cost"),
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

    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help=(
            "Timeout in seconds for ordinary API requests. "
            f"Default: {DEFAULT_REQUEST_TIMEOUT_SECONDS:g}"
        ),
    )

    parser.add_argument(
        "--comparison-timeout",
        type=float,
        default=DEFAULT_COMPARISON_TIMEOUT_SECONDS,
        help=(
            "Timeout in seconds for the provider-backed benchmark comparison. "
            f"Default: {DEFAULT_COMPARISON_TIMEOUT_SECONDS:g}"
        ),
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

    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be greater than 0")

    if args.comparison_timeout <= 0:
        raise ValueError("--comparison-timeout must be greater than 0")

    with httpx.Client(timeout=args.request_timeout) as client:
        document_id, dataset_id = get_or_create_dataset(
            client=client,
            base_url=base_url,
            policy_path=policy_path,
        )

        print("Ensuring benchmark test cases...", flush=True)

        add_test_cases(
            client=client,
            base_url=base_url,
            dataset_id=dataset_id,
            document_id=document_id,
        )

        print(
            "Ensuring default pipeline configs from API settings...",
            flush=True,
        )

        pipeline_ids = ensure_default_pipeline_configs(
            client=client,
            base_url=base_url,
        )

        print(
            "Running benchmark comparison... "
            "This executes 48 provider-backed evaluations and can take "
            "several minutes.",
            flush=True,
        )

        comparison = run_comparison(
            client=client,
            base_url=base_url,
            dataset_id=dataset_id,
            pipeline_config_ids=[
                pipeline_ids["BM25 Baseline"],
                pipeline_ids["Dense Retrieval"],
                pipeline_ids["Hybrid Retrieval"],
                pipeline_ids["Hybrid Retrieval + Rerank"],
            ],
            comparison_timeout_seconds=args.comparison_timeout,
        )

    print_summary(
        document_id=document_id,
        dataset_id=dataset_id,
        pipeline_ids=pipeline_ids,
        comparison=comparison,
    )

if __name__ == "__main__":
    main()