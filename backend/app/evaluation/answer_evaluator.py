import re
from dataclasses import dataclass
from typing import Any, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import DocumentChunk, EvaluationResult, Run, TraceStep
from backend.app.logging_config import get_logger


logger = get_logger(__name__)

EVALUATOR_TYPE = "heuristic-rag-evaluator-v1"

class EvaluationError(Exception):
    pass

@dataclass(frozen=True)
class MetricResult:
    metric_name: str
    metric_value: float
    details: dict[str, Any]

@dataclass(frozen=True)
class RunEvaluationSummary:
    run_id: str
    evaluator_type: str
    metrics: list[MetricResult]

def evaluate_rag_run(
    db: Session,
    run_id: str,
    persist: bool = True
) -> RunEvaluationSummary:
    run = db.get(Run, run_id)

    if run is None:
        raise EvaluationError(f"Run with id '{run_id}' was not found")
    if run.workflow_type != "rag_answer":
        raise EvaluationError(
            f"Run '{run_id}' has unsupported workflow_type '{run.workflow_type}'"
        )
    if run.status != "completed":
        raise EvaluationError(
            f"Run '{run_id}' must be completed before evaluation"
        )
    generated_answer = get_generated_answer_for_evaluation(
        db=db,
        run_id=run_id,
        run=run
    )

    if not generated_answer:
        raise EvaluationError(f"Run '{run_id}' has no answer to evaluate")

    retrieval_step = get_trace_step(db, run_id, "retrieval")

    retrieved_chunks_payload = retrieval_step.output_data.get("retrieved_chunks", [])
    retrieved_scores = [
        float(item.get("score", 0.0))
        for item in retrieved_chunks_payload
        if isinstance(item, dict)
    ]

    source_chunk_ids = [
        str(item.get("chunk_id"))
        for item in retrieved_chunks_payload
        if isinstance(item, dict) and item.get("chunk_id")
    ]

    source_chunks = get_source_chunks(db, source_chunk_ids)
    source_text = "\n\n".join(chunk.chunk_text for chunk in source_chunks)

    metrics = build_metrics(
        query=run.input_query,
        answer=generated_answer,
        source_text=source_text,
        source_chunks=source_chunks,
        retrieved_scores=retrieved_scores
    )

    if persist:
        save_metrics(db, run_id, metrics)

    logger.info(
        "Evaluated run %s with %s metrics",
        run_id,
        len(metrics)
    )

    return RunEvaluationSummary(
        run_id=run_id,
        evaluator_type=EVALUATOR_TYPE,
        metrics=metrics
    )

def get_trace_step(db: Session, run_id: str, step_type: str) -> TraceStep:
    trace_step = db.execute(
        select(TraceStep)
        .where(
            TraceStep.run_id == run_id,
            TraceStep.step_type == step_type
        )
        .order_by(TraceStep.step_index.asc())
    ).scalars().first()

    if trace_step is None:
        raise EvaluationError(
            f"Run '{run_id}' does not have a '{step_type}' trace step"
        )

    return trace_step

def get_source_chunks(db: Session, chunk_ids: list[str]) -> list[DocumentChunk]:
    if not chunk_ids:
        return []

    chunks = db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.id.in_(chunk_ids))
    ).scalars().all()

    chunk_by_id = {chunk.id: chunk for chunk in chunks}

    return [
        chunk_by_id[chunk_id]
        for chunk_id in chunk_ids
        if chunk_id in chunk_by_id
    ]

def get_generated_answer_for_evaluation(
    db: Session,
    run_id: str,
    run: Run
) -> Optional[str]:
    answer_step = db.execute(
        select(TraceStep)
        .where(
            TraceStep.run_id == run_id,
            TraceStep.step_type == "answer_generation"
        )
        .order_by(TraceStep.step_index.asc())
    ).scalars().first()

    if answer_step is not None:
        answer = answer_step.output_data.get("answer")

        if answer:
            return str(answer)

    return run.output_answer

def build_metrics(
    query: str,
    answer: str,
    source_text: str,
    source_chunks: list[DocumentChunk],
    retrieved_scores: list[float]
) -> list[MetricResult]:
    query_terms = extract_keywords(query)
    answer_terms = extract_keywords(answer)
    source_terms = extract_keywords(source_text)

    supported_terms = answer_terms.intersection(source_terms)
    unsupported_terms = answer_terms.difference(source_terms)

    if answer_terms:
        answer_support_score = len(supported_terms) / len(answer_terms)
    else:
        answer_support_score = 0.0

    hallucination_risk = 1.0 - answer_support_score

    matched_query_terms = query_terms.intersection(answer_terms)
    missing_query_terms = query_terms.difference(answer_terms)

    if query_terms:
        query_answer_relevance_score = len(matched_query_terms) / len(query_terms)
    else:
        query_answer_relevance_score = 0.0

    top_retrieval_score = max(retrieved_scores) if retrieved_scores else 0.0
    average_retrieval_score = (
        sum(retrieved_scores) / len(retrieved_scores)
        if retrieved_scores
        else 0.0
    )

    source_coverage_score = calculate_source_coverage_score(
        answer_terms=answer_terms,
        source_chunks=source_chunks
    )

    retrieval_confidence = clamp(top_retrieval_score, 0.0, 1.0)

    overall_quality_score = (
        0.40 * answer_support_score
        + 0.25 * query_answer_relevance_score
        + 0.20 * retrieval_confidence
        + 0.15 * source_coverage_score
    )

    return [
        MetricResult(
            metric_name="answer_support_score",
            metric_value=round(answer_support_score, 4),
            details={
                "supported_terms": sorted(supported_terms),
                "unsupported_terms": sorted(unsupported_terms),
                "answer_terms": sorted(answer_terms)
            }
        ),
        MetricResult(
            metric_name="query_answer_relevance_score",
            metric_value=round(query_answer_relevance_score, 4),
            details={
                "query_terms": sorted(query_terms),
                "matched_query_terms": sorted(matched_query_terms),
                "missing_query_terms": sorted(missing_query_terms),
                "answer_terms": sorted(answer_terms)
            }
        ),
        MetricResult(
            metric_name="hallucination_risk",
            metric_value=round(hallucination_risk, 4),
            details={
                "definition": "1 - answer_support_score",
                "unsupported_terms": sorted(unsupported_terms)
            }
        ),
        MetricResult(
            metric_name="top_retrieval_score",
            metric_value=round(top_retrieval_score, 4),
            details={
                "retrieved_scores": retrieved_scores
            }
        ),
        MetricResult(
            metric_name="average_retrieval_score",
            metric_value=round(average_retrieval_score, 4),
            details={
                "retrieved_scores": retrieved_scores
            }
        ),
        MetricResult(
            metric_name="source_coverage_score",
            metric_value=round(source_coverage_score, 4),
            details={
                "source_chunk_count": len(source_chunks)
            }
        ),
        MetricResult(
            metric_name="overall_quality_score",
            metric_value=round(overall_quality_score, 4),
            details={
                "weights": {
                    "answer_support_score": 0.40,
                    "query_answer_relevance_score": 0.25,
                    "top_retrieval_score": 0.20,
                    "source_coverage_score": 0.15
                }
            }
        ),
        MetricResult(
            metric_name="source_chunk_count",
            metric_value=float(len(source_chunks)),
            details={}
        ),
        MetricResult(
            metric_name="answer_length_chars",
            metric_value=float(len(answer)),
            details={}
        ),
    ]

def calculate_source_coverage_score(
    answer_terms: set[str],
    source_chunks: list[DocumentChunk]
) -> float:
    if not answer_terms or not source_chunks:
        return 0.0

    useful_chunks = 0

    for chunk in source_chunks:
        chunk_terms = extract_keywords(chunk.chunk_text)

        if answer_terms.intersection(chunk_terms):
            useful_chunks += 1

    return useful_chunks / len(source_chunks)

def save_metrics(
    db: Session,
    run_id: str,
    metrics: list[MetricResult]
) -> None:
    existing_results = db.execute(
        select(EvaluationResult)
        .where(
            EvaluationResult.run_id == run_id,
            EvaluationResult.evaluator_type == EVALUATOR_TYPE
        )
    ).scalars().all()

    for result in existing_results:
        db.delete(result)

    for metric in metrics:
        db.add(
            EvaluationResult(
                run_id=run_id,
                metric_name=metric.metric_name,
                metric_value=metric.metric_value,
                evaluator_type=EVALUATOR_TYPE,
                details_json=metric.details
            )
        )

    db.commit()

def extract_keywords(text: str) -> set[str]:
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "do", "does", "did","can", "could", "should", "would", 
        "i", "you", "we", "they", "he", "she", "it", "this", "that", "these", "those", "to", "for", "of","in", 
        "on", "at", "by", "with", "and", "or", "but", "from", "as","get", "be", "what", "when", "where", "who", 
        "whom", "whose", "why", "how", "tell", "me", "about", "please", "policy", "policies", "company", "may",
        "must", "does"
    }

    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())

    keywords = set()

    for token in tokens:
        normalized_token = normalize_keyword(token)

        if not normalized_token:
            continue

        if normalized_token in stop_words:
            continue

        keywords.add(normalized_token)

    return keywords


def normalize_keyword(token: str) -> str:
    synonym_map = {
        "customers": "customer",
        "customer": "customer",
        "products": "product",
        "product": "product",
        "subscriptions": "subscription",
        "subscription": "subscription",
        "cancellations": "cancellation",
        "cancellation": "cancellation",
        "payments": "payment",
        "payment": "payment",
        "charges": "charge",
        "charge": "charge",
        "days": "day",
        "day": "day",
        "hours": "hour",
        "hour": "hour",
        "times": "time",
        "time": "time",
        "usually": "usual",
        "usual": "usual",
        "reporting": "report",
        "reported": "report",
        "reports": "report",
        "report": "report",
        "damaged": "damage",
        "damage": "damage",
        "defective": "defect",
        "defects": "defect",
        "defect": "defect",
        "accessories": "accessory",
        "accessory": "accessory",
        "repairs": "repair",
        "repaired": "repair",
        "repair": "repair",
        "excludes": "cover",
        "excluded": "cover",
        "exclude": "cover",
        "exclusion": "cover",
        "coverage": "cover",
        "covered": "cover",
        "covering": "cover",
        "cover": "cover",
        "provides": "provide",
        "provided": "provide",
        "providing": "provide",
        "provide": "provide",
        "includes": "include",
        "included": "include",
        "including": "include",
        "include": "include",
        "renewals": "renewal",
        "renewal": "renewal",
        "retries": "retry",
        "retrying": "retry",
        "retry": "retry",
        "addresses": "address",
        "address": "address",
        "invoices": "invoice",
        "invoice": "invoice",
        "records": "record",
        "record": "record",
    }

    if token in synonym_map:
        return synonym_map[token]

    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"

    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]

    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]

    if len(token) > 4 and token.endswith("ly"):
        return token[:-2]

    if (
        len(token) > 4
        and token.endswith("es")
        and token.endswith(("ches", "shes", "xes", "zes", "ses"))
    ):
        return token[:-2]

    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]

    return token

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))