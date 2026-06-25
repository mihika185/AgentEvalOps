from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.connection import Base

def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_id("doc")
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="uploaded", nullable=False)

    num_pages: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    num_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now
    )

    chunks: Mapped[list[DocumentChunk]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan"
    )

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_id("chunk")
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False
    )

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)

    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    embedding_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    document: Mapped[Document] = relationship(
        "Document",
        back_populates="chunks"
    )

class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_id("exp")
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    retriever_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    llm_provider: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    llm_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    prompt_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    chunking_strategy: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    reranker_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    runs: Mapped[list[Run]] = relationship(
        "Run",
        back_populates="experiment",
        cascade="all, delete-orphan"
    )

class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_id("run")
    )
    experiment_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("experiments.id", ondelete="SET NULL"),
        nullable=True
    )

    workflow_type: Mapped[str] = mapped_column(String(50), nullable=False)
    input_query: Mapped[str] = mapped_column(Text, nullable=False)
    output_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(String(50), default="created", nullable=False)

    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    estimated_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    experiment: Mapped[Optional[Experiment]] = relationship(
        "Experiment",
        back_populates="runs"
    )
    trace_steps: Mapped[list[TraceStep]] = relationship(
        "TraceStep",
        back_populates="run",
        cascade="all, delete-orphan"
    )
    evaluation_results: Mapped[list[EvaluationResult]] = relationship(
        "EvaluationResult",
        back_populates="run",
        cascade="all, delete-orphan"
    )

class TraceStep(Base):
    __tablename__ = "trace_steps"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_id("trace")
    )
    run_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False
    )

    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    step_type: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    input_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    output_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="completed", nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    run: Mapped[Run] = relationship(
        "Run",
        back_populates="trace_steps"
    )

class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_id("eval")
    )
    run_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False
    )

    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    evaluator_type: Mapped[str] = mapped_column(String(100), nullable=False)

    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    run: Mapped[Run] = relationship(
        "Run",
        back_populates="evaluation_results"
    )

class QualityGate(Base):
    __tablename__ = "quality_gates"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_id("gate")
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    operator: Mapped[str] = mapped_column(String(10), nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

class BenchmarkDataset(Base):
    __tablename__ = "benchmark_datasets"

    id = Column(String(40), primary_key=True, default=lambda: generate_id("dataset"))
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    document_id = Column(
        String(40),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    metadata_json = Column(JSON, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now
    )


class BenchmarkTestCase(Base):
    __tablename__ = "benchmark_test_cases"

    id = Column(String(40), primary_key=True, default=lambda: generate_id("case"))

    dataset_id = Column(
        String(40),
        ForeignKey("benchmark_datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    question = Column(Text, nullable=False)

    document_id = Column(
        String(40),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    expected_behavior = Column(String(30), nullable=False)
    expected_keywords = Column(JSON, nullable=False, default=list)

    tags = Column(JSON, nullable=False, default=list)
    metadata_json = Column(JSON, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now
    )

class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id = Column(String(40), primary_key=True, default=lambda: generate_id("benchrun"))

    dataset_id = Column(
        String(40),
        ForeignKey("benchmark_datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    status = Column(String(30), nullable=False, default="running")

    total_cases = Column(Integer, nullable=False, default=0)
    passed_cases = Column(Integer, nullable=False, default=0)
    failed_cases = Column(Integer, nullable=False, default=0)

    answerable_cases = Column(Integer, nullable=False, default=0)
    answerable_passed = Column(Integer, nullable=False, default=0)
    unanswerable_cases = Column(Integer, nullable=False, default=0)
    unanswerable_passed = Column(Integer, nullable=False, default=0)

    average_answer_support_score = Column(Float, nullable=True)
    average_query_answer_relevance_score = Column(Float, nullable=True)
    average_hallucination_risk = Column(Float, nullable=True)
    average_overall_quality_score = Column(Float, nullable=True)
    average_latency_ms = Column(Float, nullable=True)

    metadata_json = Column(JSON, nullable=False, default=dict)

    started_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class BenchmarkRunItem(Base):
    __tablename__ = "benchmark_run_items"

    id = Column(String(40), primary_key=True, default=lambda: generate_id("benchitem"))

    benchmark_run_id = Column(
        String(40),
        ForeignKey("benchmark_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    test_case_id = Column(
        String(40),
        ForeignKey("benchmark_test_cases.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    rag_run_id = Column(
        String(40),
        ForeignKey("runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    question = Column(Text, nullable=False)
    expected_behavior = Column(String(30), nullable=False)
    expected_keywords = Column(JSON, nullable=False, default=list)

    actual_answer = Column(Text, nullable=True)
    passed = Column(Boolean, nullable=False, default=False)
    failure_reason = Column(Text, nullable=True)

    quality_gate_passed = Column(Boolean, nullable=False, default=False)
    response_blocked_by_quality_gate = Column(Boolean, nullable=False, default=False)

    metrics_json = Column(JSON, nullable=False, default=dict)
    source_chunks_json = Column(JSON, nullable=False, default=list)
    latency_ms = Column(Integer, nullable=True)

    metadata_json = Column(JSON, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)