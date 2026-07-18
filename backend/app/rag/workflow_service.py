import time
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database.models import Document
from backend.app.evaluation.answer_evaluator import MetricResult, evaluate_rag_run
from backend.app.evaluation.cost_latency_tracker import track_answer_generation_usage
from backend.app.evaluation.quality_gates import (
    DEFAULT_QUALITY_GATE_PROFILE,
    QualityGateError,
    evaluate_quality_gates,
    normalize_quality_gate_profile_name,
)
from backend.app.logging_config import get_logger
from backend.app.observability.run_recorder import (
    RunRecorderError,
    complete_run,
    create_run,
    fail_run,
    record_trace_step,
)
from backend.app.rag.answer_service import AnswerGenerator, SourceChunk
from backend.app.rag.citation_checker import (
    Citation,
    CitationCheckResult,
    citation_to_dict,
    check_answer_citations,
)
from backend.app.rag.llm_answer_generator import (
    LLMAnswerGenerationError,
    get_default_answer_generator,
)
from backend.app.retrieval.retrieval_service import RetrievalError, retrieve_chunks


logger = get_logger(__name__)


class RAGWorkflowError(Exception):
    pass

@dataclass(frozen=True)
class ObservableRAGAnswerResult:
    run_id: str
    query: str
    answer: str
    retrieval_provider: str
    source_chunks: list[SourceChunk]
    citations: list[Citation]
    citation_check_passed: bool
    citation_accuracy_score: float
    citation_failed_reasons: list[str]
    retrieval_top_k: int
    retrieved_chunk_count: int
    document_id: Optional[str]
    answer_generator: str
    reranker_used: bool
    reranker_name: Optional[str]
    total_latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    evaluation_metrics: list[MetricResult]
    quality_gate_profile: str
    quality_gate_passed: bool
    quality_gate_pass_rate: float
    failed_quality_gates: list[str]
    response_blocked_by_quality_gate: bool

def run_rag_answer_workflow(
    db: Session,
    query: str,
    top_k: int = settings.default_retrieval_top_k,
    document_id: Optional[str] = None,
    retrieval_provider: str = "dense",
    answer_generator: Optional[AnswerGenerator] = None,
    quality_gate_profile: str = DEFAULT_QUALITY_GATE_PROFILE,
    rerank: bool = False,
    candidate_multiplier: int = 3,
    experiment_id: Optional[str] = None,
) -> ObservableRAGAnswerResult:
    cleaned_query = query.strip()

    if not cleaned_query:
        raise RAGWorkflowError("Query cannot be empty")

    if top_k <= 0:
        raise RAGWorkflowError("top_k must be greater than 0")

    if top_k > settings.max_retrieval_top_k:
        raise RAGWorkflowError(
            f"top_k cannot be greater than {settings.max_retrieval_top_k}"
        )

    if candidate_multiplier <= 0:
        raise RAGWorkflowError("candidate_multiplier must be greater than 0")

    if candidate_multiplier > 10:
        raise RAGWorkflowError("candidate_multiplier cannot be greater than 10")

    resolved_retrieval_provider = normalize_retrieval_provider(retrieval_provider)

    try:
        resolved_quality_gate_profile = normalize_quality_gate_profile_name(
            quality_gate_profile
        )
    except QualityGateError as exc:
        raise RAGWorkflowError(str(exc)) from exc

    total_start = time.perf_counter()
    run_id: Optional[str] = None

    try:
        run = create_run(
            db=db,
            workflow_type="rag_answer",
            input_query=cleaned_query,
            experiment_id=experiment_id,
            metadata={
                "document_id": document_id,
                "top_k": top_k,
                "retrieval_provider": resolved_retrieval_provider,
                "quality_gate_profile": resolved_quality_gate_profile,
                "rerank": rerank,
                "candidate_multiplier": candidate_multiplier,
                "workflow_version": "rag-answer-v2-citations",
            },
        )

        run_id = run.run_id

        if document_id is not None and db.get(Document, document_id) is None:
            error_message = f"Document with id '{document_id}' was not found"

            record_trace_step(
                db=db,
                run_id=run_id,
                step_index=0,
                step_type="validation",
                name="validate_document_scope",
                input_data={
                    "document_id": document_id,
                    "retrieval_provider": resolved_retrieval_provider,
                    "rerank": rerank,
                    "candidate_multiplier": candidate_multiplier,
                },
                output_data={},
                status="failed",
                error_message=error_message,
            )

            fail_run(
                db=db,
                run_id=run_id,
                error_message=error_message,
                latency_ms=elapsed_ms(total_start),
                metadata={
                    "failure_stage": "document_validation",
                },
            )

            raise RAGWorkflowError(error_message)

        retrieval_start = time.perf_counter()

        retrieval_result = retrieve_chunks(
            query=cleaned_query,
            top_k=top_k,
            document_id=document_id,
            method=resolved_retrieval_provider,
            db=db,
            rerank=rerank,
            candidate_multiplier=candidate_multiplier,
        )

        retrieval_latency_ms = elapsed_ms(retrieval_start)

        record_trace_step(
            db=db,
            run_id=run_id,
            step_index=0,
            step_type="retrieval",
            name="retrieve_relevant_chunks",
            input_data={
                "query": cleaned_query,
                "top_k": top_k,
                "document_id": document_id,
                "retrieval_provider": resolved_retrieval_provider,
                "rerank": rerank,
                "candidate_multiplier": candidate_multiplier,
            },
            output_data={
                "retrieved_chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "score": chunk.score,
                        "metadata": chunk.metadata,
                    }
                    for chunk in retrieval_result.chunks
                ],
                "requested_top_k": top_k,
                "returned_chunk_count": len(retrieval_result.chunks),
                "reranker_used": retrieval_result.reranker_used,
                "reranker_name": retrieval_result.reranker_name,
                "retrieved_count": len(retrieval_result.chunks),
                "retrieval_method": retrieval_result.retrieval_method,
                "collection_name": retrieval_result.collection_name,
                "embedding_provider": retrieval_result.embedding_provider,
                "embedding_model": retrieval_result.embedding_model,
            },
            latency_ms=retrieval_latency_ms,
        )

        generator = answer_generator or get_default_answer_generator()

        answer_start = time.perf_counter()

        answer = generator.generate_answer(
            query=cleaned_query,
            chunks=retrieval_result.chunks,
        )

        answer_latency_ms = elapsed_ms(answer_start)

        source_chunks = [
            SourceChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                score=chunk.score,
                text=chunk.text,
                metadata=chunk.metadata,
            )
            for chunk in retrieval_result.chunks
        ]

        generation_usage = track_answer_generation_usage(
            generator=generator,
            generator_name=generator.generator_name,
            query=cleaned_query,
            source_chunks=source_chunks,
            answer=answer,
            latency_ms=answer_latency_ms,
        )

        generation_retry_metadata = get_generation_retry_metadata(
            generator
        )

        record_trace_step(
            db=db,
            run_id=run_id,
            step_index=1,
            step_type="answer_generation",
            name=generator.generator_name,
            input_data={
                "query": cleaned_query,
                "source_chunk_ids": [chunk.chunk_id for chunk in source_chunks],
                "source_chunk_count": len(source_chunks),
            },
            output_data={
                "answer": answer,
                "answer_length": len(answer),
                "token_usage": generation_usage.to_metadata(),
                "retry_metadata": generation_retry_metadata,
            },
            latency_ms=answer_latency_ms,
        )

        citation_start = time.perf_counter()

        citation_check = check_answer_citations(
            answer=answer,
            source_chunks=source_chunks,
        )

        citation_latency_ms = elapsed_ms(citation_start)

        record_citation_trace_step(
            db=db,
            run_id=run_id,
            citation_check=citation_check,
            latency_ms=citation_latency_ms,
        )

        total_latency_ms = elapsed_ms(total_start)

        complete_run(
            db=db,
            run_id=run_id,
            output_answer=answer,
            latency_ms=total_latency_ms,
            prompt_tokens=generation_usage.prompt_tokens,
            completion_tokens=generation_usage.completion_tokens,
            estimated_cost=generation_usage.estimated_cost,
            metadata={
                "document_id": document_id,
                "retrieval_top_k": top_k,
                "retrieval_provider": retrieval_result.retrieval_method,
                "source_chunk_count": len(source_chunks),
                "answer_generator": generator.generator_name,
                "retrieval_latency_ms": retrieval_latency_ms,
                "answer_generation_latency_ms": answer_latency_ms,
                "prompt_tokens": generation_usage.prompt_tokens,
                "completion_tokens": generation_usage.completion_tokens,
                "total_tokens": generation_usage.total_tokens,
                "estimated_cost": generation_usage.estimated_cost,
                "token_usage_source": generation_usage.token_usage_source,
                "token_usage": generation_usage.to_metadata(),
                "answer_generation_retry": generation_retry_metadata,
                "citation_check_latency_ms": citation_latency_ms,
                "citation_check_passed": citation_check.citation_check_passed,
                "citation_accuracy_score": citation_check.citation_accuracy_score,
                "citation_count": citation_check.total_citation_count,
                "cited_chunk_ids": citation_check.cited_chunk_ids,
                "rerank": rerank,
                "candidate_multiplier": candidate_multiplier,
                "retrieved_chunk_count": len(source_chunks),
                "reranker_used": retrieval_result.reranker_used,
                "reranker_name": retrieval_result.reranker_name,
            },
        )

        evaluation_start = time.perf_counter()

        evaluation_summary = evaluate_rag_run(
            db=db,
            run_id=run_id,
            persist=True,
        )

        evaluation_latency_ms = elapsed_ms(evaluation_start)

        record_trace_step(
            db=db,
            run_id=run_id,
            step_index=3,
            step_type="evaluation",
            name=evaluation_summary.evaluator_type,
            input_data={
                "run_id": run_id,
            },
            output_data={
                "metrics": [
                    {
                        "metric_name": metric.metric_name,
                        "metric_value": metric.metric_value,
                        "details": metric.details,
                    }
                    for metric in evaluation_summary.metrics
                ],
                "metric_count": len(evaluation_summary.metrics),
            },
            latency_ms=evaluation_latency_ms,
        )

        quality_gate_start = time.perf_counter()

        quality_gate_summary = evaluate_quality_gates(
            db=db,
            run_id=run_id,
            persist=True,
            profile_name=resolved_quality_gate_profile,
        )

        quality_gate_latency_ms = elapsed_ms(quality_gate_start)

        failed_quality_gates = [
            check.gate_name
            for check in quality_gate_summary.checks
            if not check.passed
        ]

        record_trace_step(
            db=db,
            run_id=run_id,
            step_index=4,
            step_type="quality_gate_evaluation",
            name=f"evaluate_quality_gates:{quality_gate_summary.profile_name}",
            input_data={
                "quality_gate_profile": quality_gate_summary.profile_name,
                "quality_gate_passed": quality_gate_summary.overall_passed,
                "failed_quality_gates": failed_quality_gates,
            },
            output_data={
                "overall_passed": quality_gate_summary.overall_passed,
                "passed_count": quality_gate_summary.passed_count,
                "failed_count": quality_gate_summary.failed_count,
                "total_gates": quality_gate_summary.total_gates,
                "pass_rate": quality_gate_summary.pass_rate,
                "checks": [
                    {
                        "gate_id": check.gate_id,
                        "gate_name": check.gate_name,
                        "metric_name": check.metric_name,
                        "metric_value": check.metric_value,
                        "operator": check.operator,
                        "threshold": check.threshold,
                        "passed": check.passed,
                    }
                    for check in quality_gate_summary.checks
                ],
            },
            latency_ms=quality_gate_latency_ms,
        )

        response_blocked_by_quality_gate = not quality_gate_summary.overall_passed

        final_answer = answer
        final_citations = citation_check.citations
        final_citation_check_passed = citation_check.citation_check_passed
        final_citation_accuracy_score = citation_check.citation_accuracy_score
        final_citation_failed_reasons = citation_check.failed_reasons

        if response_blocked_by_quality_gate:
            final_answer = build_quality_gate_fallback_answer()
            final_citations = []
            final_citation_check_passed = False
            final_citation_accuracy_score = 0.0
            final_citation_failed_reasons = [
                *citation_check.failed_reasons,
                "response_blocked_by_quality_gate",
            ]

        record_trace_step(
            db=db,
            run_id=run_id,
            step_index=5,
            step_type="response_finalization",
            name="apply_quality_gate_response_policy",
            input_data={
                "quality_gate_passed": quality_gate_summary.overall_passed,
                "failed_quality_gates": failed_quality_gates,
                "raw_citation_check_passed": citation_check.citation_check_passed,
                "raw_citation_accuracy_score": citation_check.citation_accuracy_score,
            },
            output_data={
                "raw_generated_answer": answer,
                "final_answer": final_answer,
                "response_blocked_by_quality_gate": response_blocked_by_quality_gate,
                "final_citations": [
                    citation_to_dict(citation)
                    for citation in final_citations
                ],
                "final_citation_check_passed": final_citation_check_passed,
                "final_citation_accuracy_score": final_citation_accuracy_score,
            },
            latency_ms=0,
        )

        total_latency_ms = elapsed_ms(total_start)

        complete_run(
            db=db,
            run_id=run_id,
            output_answer=final_answer,
            latency_ms=total_latency_ms,
            prompt_tokens=generation_usage.prompt_tokens,
            completion_tokens=generation_usage.completion_tokens,
            estimated_cost=generation_usage.estimated_cost,
            metadata={
                "document_id": document_id,
                "retrieval_top_k": top_k,
                "retrieval_provider": retrieval_result.retrieval_method,
                "source_chunk_count": len(source_chunks),
                "answer_generator": generator.generator_name,
                "retrieval_latency_ms": retrieval_latency_ms,
                "answer_generation_latency_ms": answer_latency_ms,
                "prompt_tokens": generation_usage.prompt_tokens,
                "completion_tokens": generation_usage.completion_tokens,
                "total_tokens": generation_usage.total_tokens,
                "estimated_cost": generation_usage.estimated_cost,
                "token_usage_source": generation_usage.token_usage_source,
                "token_usage": generation_usage.to_metadata(),
                "answer_generation_retry": generation_retry_metadata,
                "citation_check_latency_ms": citation_latency_ms,
                "evaluation_latency_ms": evaluation_latency_ms,
                "quality_gate_latency_ms": quality_gate_latency_ms,
                "raw_citation_check_passed": citation_check.citation_check_passed,
                "raw_citation_accuracy_score": citation_check.citation_accuracy_score,
                "citation_check_passed": final_citation_check_passed,
                "citation_accuracy_score": final_citation_accuracy_score,
                "citation_count": len(final_citations),
                "cited_chunk_ids": [
                    citation.chunk_id
                    for citation in final_citations
                ],
                "citation_failed_reasons": final_citation_failed_reasons,
                "quality_gate_passed": quality_gate_summary.overall_passed,
                "quality_gate_pass_rate": quality_gate_summary.pass_rate,
                "failed_quality_gates": failed_quality_gates,
                "raw_generated_answer": answer,
                "response_blocked_by_quality_gate": response_blocked_by_quality_gate,
                "quality_gate_profile": quality_gate_summary.profile_name,
                "rerank": rerank,
                "candidate_multiplier": candidate_multiplier,
                "retrieved_chunk_count": len(source_chunks),
                "reranker_used": retrieval_result.reranker_used,
                "reranker_name": retrieval_result.reranker_name,
            },
        )

        logger.info(
            "Completed observable RAG workflow %s in %sms",
            run_id,
            total_latency_ms,
        )

        return ObservableRAGAnswerResult(
            run_id=run_id,
            query=cleaned_query,
            answer=final_answer,
            retrieval_provider=retrieval_result.retrieval_method,
            source_chunks=source_chunks,
            citations=final_citations,
            citation_check_passed=final_citation_check_passed,
            citation_accuracy_score=final_citation_accuracy_score,
            citation_failed_reasons=final_citation_failed_reasons,
            retrieval_top_k=top_k,
            document_id=document_id,
            answer_generator=generator.generator_name,
            total_latency_ms=total_latency_ms,
            prompt_tokens=generation_usage.prompt_tokens,
            completion_tokens=generation_usage.completion_tokens,
            total_tokens=generation_usage.total_tokens,
            estimated_cost=generation_usage.estimated_cost,
            evaluation_metrics=evaluation_summary.metrics,
            quality_gate_passed=quality_gate_summary.overall_passed,
            quality_gate_pass_rate=quality_gate_summary.pass_rate,
            failed_quality_gates=failed_quality_gates,
            response_blocked_by_quality_gate=response_blocked_by_quality_gate,
            quality_gate_profile=quality_gate_summary.profile_name,
            retrieved_chunk_count=len(source_chunks),
            reranker_used=retrieval_result.reranker_used,
            reranker_name=retrieval_result.reranker_name,
        )

    except (
        RetrievalError,
        RunRecorderError,
        QualityGateError,
        LLMAnswerGenerationError,
    ) as exc:
        mark_run_failed_safely(
            db=db,
            run_id=run_id,
            error_message=str(exc),
            latency_ms=elapsed_ms(total_start),
        )

        raise RAGWorkflowError(str(exc)) from exc

    except RAGWorkflowError:
        raise

    except Exception as exc:
        logger.exception("Unexpected observable RAG workflow failure")

        mark_run_failed_safely(
            db=db,
            run_id=run_id,
            error_message="Failed to run RAG answer workflow",
            latency_ms=elapsed_ms(total_start),
        )

        raise RAGWorkflowError("Failed to run RAG answer workflow") from exc

def record_citation_trace_step(
    db: Session,
    run_id: str,
    citation_check: CitationCheckResult,
    latency_ms: int,
) -> None:
    record_trace_step(
        db=db,
        run_id=run_id,
        step_index=2,
        step_type="citation_check",
        name="validate_structured_citations",
        input_data={
            "max_citations": 3,
            "minimum_support_score": 0.20,
        },
        output_data={
            "citations": [
                citation_to_dict(citation)
                for citation in citation_check.citations
            ],
            "citation_check_passed": citation_check.citation_check_passed,
            "citation_accuracy_score": citation_check.citation_accuracy_score,
            "failed_reasons": citation_check.failed_reasons,
            "cited_chunk_ids": citation_check.cited_chunk_ids,
            "valid_citation_count": citation_check.valid_citation_count,
            "total_citation_count": citation_check.total_citation_count,
        },
        latency_ms=latency_ms,
    )

def get_generation_retry_metadata(
    generator: AnswerGenerator,
) -> dict[str, object]:
    return {
        "retry_count": int(
            getattr(generator, "last_retry_count", 0) or 0
        ),
        "retry_delays_seconds": list(
            getattr(
                generator,
                "last_retry_delays_seconds",
                [],
            ) or []
        ),
        "retry_reasons": list(
            getattr(generator, "last_retry_reasons", []) or []
        ),
    }

def normalize_retrieval_provider(provider: str) -> str:
    cleaned_provider = provider.strip().lower()

    if cleaned_provider in {"dense", "qdrant_vector_search", "vector"}:
        return "dense"

    if cleaned_provider in {"bm25", "keyword"}:
        return "bm25"

    if cleaned_provider in {"hybrid", "hybrid_retrieval"}:
        return "hybrid"

    raise RAGWorkflowError(f"Unsupported retrieval provider: {provider}")

def elapsed_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)

def mark_run_failed_safely(
    db: Session,
    run_id: Optional[str],
    error_message: str,
    latency_ms: int,
) -> None:
    if run_id is None:
        return

    try:
        fail_run(
            db=db,
            run_id=run_id,
            error_message=error_message,
            latency_ms=latency_ms,
        )
    except Exception:
        logger.exception("Failed to mark run as failed: %s", run_id)

def build_quality_gate_fallback_answer() -> str:
    return (
        "I could not find enough reliable evidence in the provided documents "
        "to answer this confidently."
    )