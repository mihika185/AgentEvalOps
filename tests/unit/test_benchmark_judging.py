from backend.app.api.benchmarks import judge_benchmark_case


def test_answerable_case_passes_when_answer_is_allowed_and_keywords_are_present():
    passed, failure_reason = judge_benchmark_case(
        expected_behavior="answerable",
        expected_keywords=["subscription", "billing cycle"],
        answer="Subscription cancellations take effect at the end of the billing cycle.",
        quality_gate_passed=True,
        response_blocked_by_quality_gate=False
    )

    assert passed is True
    assert failure_reason is None


def test_answerable_case_fails_when_response_is_blocked():
    passed, failure_reason = judge_benchmark_case(
        expected_behavior="answerable",
        expected_keywords=["subscription"],
        answer="I could not find enough reliable evidence in the provided documents to answer this confidently.",
        quality_gate_passed=False,
        response_blocked_by_quality_gate=True
    )

    assert passed is False
    assert failure_reason == "Expected answerable response, but quality gate blocked the answer"


def test_answerable_case_fails_when_expected_keywords_are_missing():
    passed, failure_reason = judge_benchmark_case(
        expected_behavior="answerable",
        expected_keywords=["billing cycle"],
        answer="Subscription cancellations are mentioned in the document.",
        quality_gate_passed=True,
        response_blocked_by_quality_gate=False
    )

    assert passed is False
    assert failure_reason == "Answer is missing expected keywords: billing cycle"


def test_unanswerable_case_passes_when_response_is_blocked():
    passed, failure_reason = judge_benchmark_case(
        expected_behavior="unanswerable",
        expected_keywords=[],
        answer="I could not find enough reliable evidence in the provided documents to answer this confidently.",
        quality_gate_passed=False,
        response_blocked_by_quality_gate=True
    )

    assert passed is True
    assert failure_reason is None


def test_unanswerable_case_fails_when_answer_is_returned():
    passed, failure_reason = judge_benchmark_case(
        expected_behavior="unanswerable",
        expected_keywords=[],
        answer="The company offers space travel insurance.",
        quality_gate_passed=True,
        response_blocked_by_quality_gate=False
    )

    assert passed is False
    assert failure_reason == "Expected unanswerable query to be blocked, but answer was returned"