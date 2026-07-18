import httpx
import pytest

from scripts import seed_apexcart_benchmark as seed_module
from scripts.seed_apexcart_benchmark import (
    TEST_CASES,
    add_test_cases,
    build_test_case_payload,
    request_json,
    resolve_relevant_chunk_ids,
    run_comparison,
)

def make_ground_truth_chunks():
    chunks = []

    for index, test_case in enumerate(TEST_CASES):
        markers = test_case.get("relevant_text_markers") or []

        if not markers:
            continue

        chunks.append(
            {
                "id": f"chunk_{index}",
                "chunk_text": " ".join(markers),
            }
        )

    return chunks

def test_all_answerable_cases_resolve_retrieval_ground_truth():
    chunks = make_ground_truth_chunks()

    for test_case in TEST_CASES:
        relevant_chunk_ids = resolve_relevant_chunk_ids(test_case, chunks)

        if test_case["expected_behavior"] == "answerable":
            assert relevant_chunk_ids
        else:
            assert relevant_chunk_ids == []

def test_resolve_relevant_chunk_ids_rejects_missing_marker():
    test_case = {
        "question": "Question",
        "relevant_text_markers": ["missing evidence"],
    }

    with pytest.raises(RuntimeError, match="Missing marker"):
        resolve_relevant_chunk_ids(
            test_case=test_case,
            chunks=[{"id": "chunk_1", "chunk_text": "different evidence"}],
        )

def test_build_test_case_payload_adds_auditable_ground_truth_metadata():
    test_case = TEST_CASES[0]
    payload = build_test_case_payload(test_case, make_ground_truth_chunks())

    assert "relevant_text_markers" not in payload
    assert payload["metadata_json"]["expected_relevant_chunk_ids"] == ["chunk_0"]
    assert payload["metadata_json"]["ground_truth_method"] == (
        "exact_text_marker_match_v1"
    )
    assert payload["metadata_json"]["ground_truth_markers"] == (
        test_case["relevant_text_markers"]
    )

def test_add_test_cases_updates_existing_cases_and_creates_missing_cases(monkeypatch):
    requests = []
    existing_question = TEST_CASES[0]["question"]

    monkeypatch.setattr(
        seed_module,
        "list_document_chunks",
        lambda **kwargs: make_ground_truth_chunks(),
    )
    monkeypatch.setattr(
        seed_module,
        "list_test_cases",
        lambda **kwargs: [
            {
                "id": "case_existing",
                "question": existing_question,
            }
        ],
    )

    def fake_request_json(client, method, url, **kwargs):
        requests.append(
            {
                "method": method,
                "url": url,
                "json": kwargs.get("json"),
            }
        )
        return {"id": "created"}

    monkeypatch.setattr(seed_module, "request_json", fake_request_json)

    add_test_cases(
        client=object(),
        base_url="http://localhost:8000",
        dataset_id="dataset_test",
        document_id="document_test",
    )

    patch_requests = [request for request in requests if request["method"] == "PATCH"]
    post_requests = [request for request in requests if request["method"] == "POST"]

    assert len(patch_requests) == 1
    assert patch_requests[0]["url"].endswith(
        "/api/v1/benchmarks/test-cases/case_existing"
    )
    assert len(post_requests) == len(TEST_CASES) - 1
    assert all(
        "expected_relevant_chunk_ids" in request["json"]["metadata_json"]
        for request in requests
    )

def test_run_comparison_uses_dedicated_timeout(monkeypatch):
    captured = {}

    def fake_request_json(
        client,
        method,
        url,
        request_timeout=None,
        **kwargs,
    ):
        captured["method"] = method
        captured["url"] = url
        captured["request_timeout"] = request_timeout
        captured["json"] = kwargs.get("json")

        return {"results": []}

    monkeypatch.setattr(seed_module, "request_json", fake_request_json)

    result = run_comparison(
        client=object(),
        base_url="http://localhost:8000",
        dataset_id="dataset_test",
        pipeline_config_ids=["pipeline_1"],
        comparison_timeout_seconds=900.0,
    )

    assert result == {"results": []}
    assert captured["method"] == "POST"
    assert captured["url"].endswith(
        "/api/v1/benchmarks/datasets/dataset_test/compare"
    )
    assert captured["request_timeout"] == 900.0
    assert captured["json"] == {
        "pipeline_config_ids": ["pipeline_1"]
    }

def test_request_json_reports_read_timeout_without_blind_retry():
    request = httpx.Request(
        "POST",
        "http://localhost:8000/compare",
    )

    class TimeoutClient:
        timeout = httpx.Timeout(120.0)

        def request(self, *args, **kwargs):
            raise httpx.ReadTimeout(
                "timed out",
                request=request,
            )

    with pytest.raises(
        RuntimeError,
        match="inspect recent benchmark runs before retrying",
    ):
        request_json(
            client=TimeoutClient(),
            method="POST",
            url="http://localhost:8000/compare",
            request_timeout=900.0,
        )