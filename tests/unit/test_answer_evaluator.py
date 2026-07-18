from backend.app.evaluation.answer_evaluator import build_metrics


def metrics_by_name(metrics):
    return {
        metric.metric_name: metric.metric_value
        for metric in metrics
    }


def test_answerable_response_has_high_support_and_query_relevance():
    metrics = build_metrics(
        query="When do subscription cancellations take effect?",
        answer="Subscription cancellations take effect at the end of the billing cycle.",
        source_text=(
            "Subscription cancellations take effect at the end of the billing cycle."
        ),
        source_chunks=[],
        retrieved_scores=[0.42]
    )

    values = metrics_by_name(metrics)

    assert values["answer_support_score"] == 1.0
    assert values["query_answer_relevance_score"] == 1.0
    assert values["hallucination_risk"] == 0.0
    assert values["top_retrieval_score"] == 0.42


def test_unrelated_answer_has_low_query_relevance():
    metrics = build_metrics(
        query="Does the company offer space travel insurance?",
        answer="Customers can request a refund within 7 days of purchase.",
        source_text="Customers can request a refund within 7 days of purchase.",
        source_chunks=[],
        retrieved_scores=[0.28]
    )

    values = metrics_by_name(metrics)

    assert values["answer_support_score"] == 1.0
    assert values["query_answer_relevance_score"] == 0.0
    assert values["top_retrieval_score"] == 0.28


def test_unsupported_answer_increases_hallucination_risk():
    metrics = build_metrics(
        query="When do subscription cancellations take effect?",
        answer="Subscription cancellations include free space travel insurance.",
        source_text="Subscription cancellations take effect at the end of the billing cycle.",
        source_chunks=[],
        retrieved_scores=[0.40]
    )

    values = metrics_by_name(metrics)

    assert values["answer_support_score"] < 1.0
    assert values["hallucination_risk"] > 0.0

def test_exclusion_answer_passes_relevance_with_subject_and_intent_match():
    answer = (
        "The warranty excludes water damage, accidental drops, misuse, "
        "normal wear and tear, cosmetic scratches, unauthorized repairs, "
        "or damage caused by third-party chargers and cables."
    )

    metrics = build_metrics(
        query="What does the hardware accessory warranty exclude?",
        answer=answer,
        source_text=answer,
        source_chunks=[],
        retrieved_scores=[1.0],
        citation_accuracy_score=1.0,
    )

    values = metrics_by_name(metrics)

    assert values["answer_support_score"] == 1.0
    assert values["query_answer_relevance_score"] >= 0.6
    assert values["hallucination_risk"] == 0.0


def test_exclusion_signal_does_not_rescue_unrelated_answer():
    metrics = build_metrics(
        query="What does the hardware accessory warranty exclude?",
        answer="The subscription excludes cryptocurrency payments.",
        source_text="The subscription excludes cryptocurrency payments.",
        source_chunks=[],
        retrieved_scores=[1.0],
        citation_accuracy_score=1.0,
    )

    values = metrics_by_name(metrics)

    assert values["query_answer_relevance_score"] < 0.6