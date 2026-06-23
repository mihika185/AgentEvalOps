import time
from dataclasses import dataclass
from typing import Optional
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.logging_config import get_logger
from backend.app.observability.run_recorder import (
    RunRecorderError,
    complete_run,
    create_run,
    fail_run,
    record_trace_step,
)
from backend.app.rag.answer_service import (
    AnswerGenerator,
    SimpleExtractiveAnswerGenerator,
    SourceChunk,
)
from backend.app.retrieval.retrieval_service import RetrievalError, retrieve_relevant_chunks


logger = get_logger(__name__)


class RAGWorkflowError(Exception):
    pass


@dataclass(frozen=True)
class ObservableRAGAnswerResult:
    run_id: str
    query: str
    answer: str
    source_chunks: list[SourceChunk]
    retrieval_top_k: int
    document_id: Optional[str]
    answer_generator: str
    total_latency_ms: int


def run_rag_answer_workflow(
    db: Session,
    query: str,
    top_k: int = settings.default_retrieval_top_k,
    document_id: Optional[str] = None,
    answer_generator: Optional[AnswerGenerator] = None
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

    total_start = time.perf_counter()
    run_id: Optional[str] = None

    try:
        run = create_run(
            db=db,
            workflow_type="rag_answer",
            input_query=cleaned_query,
            metadata={
                "document_id": document_id,
                "top_k": top_k,
                "workflow_version": "rag-answer-v1"
            }
        )

        run_id = run.run_id

        retrieval_start = time.perf_counter()

        retrieval_result = retrieve_relevant_chunks(
            query=cleaned_query,
            top_k=top_k,
            document_id=document_id
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
                "document_id": document_id
            },
            output_data={
                "retrieved_chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "score": chunk.score,
                        "metadata": chunk.metadata
                    }
                    for chunk in retrieval_result.chunks
                ],
                "retrieved_count": len(retrieval_result.chunks),
                "collection_name": retrieval_result.collection_name,
                "embedding_provider": retrieval_result.embedding_provider,
                "embedding_model": retrieval_result.embedding_model
            },
            latency_ms=retrieval_latency_ms
        )

        generator = answer_generator or SimpleExtractiveAnswerGenerator()

        answer_start = time.perf_counter()

        answer = generator.generate_answer(
            query=cleaned_query,
            chunks=retrieval_result.chunks
        )

        answer_latency_ms = elapsed_ms(answer_start)

        source_chunks = [
            SourceChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                score=chunk.score,
                text=chunk.text,
                metadata=chunk.metadata
            )
            for chunk in retrieval_result.chunks
        ]

        record_trace_step(
            db=db,
            run_id=run_id,
            step_index=1,
            step_type="answer_generation",
            name=generator.generator_name,
            input_data={
                "query": cleaned_query,
                "source_chunk_ids": [chunk.chunk_id for chunk in source_chunks],
                "source_chunk_count": len(source_chunks)
            },
            output_data={
                "answer": answer,
                "answer_length": len(answer)
            },
            latency_ms=answer_latency_ms
        )

        total_latency_ms = elapsed_ms(total_start)

        complete_run(
            db=db,
            run_id=run_id,
            output_answer=answer,
            latency_ms=total_latency_ms,
            metadata={
                "document_id": document_id,
                "retrieval_top_k": top_k,
                "source_chunk_count": len(source_chunks),
                "answer_generator": generator.generator_name,
                "retrieval_latency_ms": retrieval_latency_ms,
                "answer_generation_latency_ms": answer_latency_ms
            }
        )

        logger.info(
            "Completed observable RAG workflow %s in %sms",
            run_id,
            total_latency_ms
        )

        return ObservableRAGAnswerResult(
            run_id=run_id,
            query=cleaned_query,
            answer=answer,
            source_chunks=source_chunks,
            retrieval_top_k=top_k,
            document_id=document_id,
            answer_generator=generator.generator_name,
            total_latency_ms=total_latency_ms
        )

    except (RetrievalError, RunRecorderError) as exc:
        mark_run_failed_safely(
            db=db,
            run_id=run_id,
            error_message=str(exc),
            latency_ms=elapsed_ms(total_start)
        )

        raise RAGWorkflowError(str(exc)) from exc

    except Exception as exc:
        logger.exception("Unexpected observable RAG workflow failure")

        mark_run_failed_safely(
            db=db,
            run_id=run_id,
            error_message="Failed to run RAG answer workflow",
            latency_ms=elapsed_ms(total_start)
        )

        raise RAGWorkflowError("Failed to run RAG answer workflow") from exc


def elapsed_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def mark_run_failed_safely(
    db: Session,
    run_id: Optional[str],
    error_message: str,
    latency_ms: int
) -> None:
    if run_id is None:
        return

    try:
        fail_run(
            db=db,
            run_id=run_id,
            error_message=error_message,
            latency_ms=latency_ms
        )
    except Exception:
        logger.exception("Failed to mark run as failed: %s", run_id)