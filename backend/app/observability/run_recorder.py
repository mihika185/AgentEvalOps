from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from sqlalchemy.orm import Session

from backend.app.database.models import Run, TraceStep
from backend.app.logging_config import get_logger

logger = get_logger(__name__)

class RunRecorderError(Exception):
    pass

@dataclass(frozen=True)
class RecordedRun:
    run_id: str
    workflow_type: str
    status: str

@dataclass(frozen=True)
class RecordedTraceStep:
    trace_step_id: str
    run_id: str
    step_index: int
    step_type: str
    name: str
    status: str

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def create_run(
    db: Session,
    workflow_type: str,
    input_query: str,
    experiment_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None
) -> RecordedRun:
    if not workflow_type.strip():
        raise RunRecorderError("workflow_type cannot be empty")

    if not input_query.strip():
        raise RunRecorderError("input_query cannot be empty")

    run = Run(
        experiment_id=experiment_id,
        workflow_type=workflow_type.strip(),
        input_query=input_query.strip(),
        status="running",
        metadata_json=metadata or {}
    )

    db.add(run)
    db.commit()
    db.refresh(run)

    logger.info("Created run: %s", run.id)

    return RecordedRun(
        run_id=run.id,
        workflow_type=run.workflow_type,
        status=run.status
    )

def record_trace_step(
    db: Session,
    run_id: str,
    step_index: int,
    step_type: str,
    name: str,
    input_data: Optional[dict[str, Any]] = None,
    output_data: Optional[dict[str, Any]] = None,
    latency_ms: Optional[int] = None,
    status: str = "completed",
    error_message: Optional[str] = None
) -> RecordedTraceStep:
    if step_index < 0:
        raise RunRecorderError("step_index cannot be negative")

    if not step_type.strip():
        raise RunRecorderError("step_type cannot be empty")

    if not name.strip():
        raise RunRecorderError("name cannot be empty")

    run = db.get(Run, run_id)

    if run is None:
        raise RunRecorderError(f"Run with id '{run_id}' was not found")

    trace_step = TraceStep(
        run_id=run_id,
        step_index=step_index,
        step_type=step_type.strip(),
        name=name.strip(),
        input_data=input_data or {},
        output_data=output_data or {},
        latency_ms=latency_ms,
        status=status,
        error_message=error_message
    )

    db.add(trace_step)
    db.commit()
    db.refresh(trace_step)

    logger.info(
        "Recorded trace step %s for run %s",
        trace_step.name,
        run_id
    )

    return RecordedTraceStep(
        trace_step_id=trace_step.id,
        run_id=trace_step.run_id,
        step_index=trace_step.step_index,
        step_type=trace_step.step_type,
        name=trace_step.name,
        status=trace_step.status
    )

def complete_run(
    db: Session,
    run_id: str,
    output_answer: str,
    latency_ms: Optional[int] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    estimated_cost: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None
) -> RecordedRun:
    run = db.get(Run, run_id)

    if run is None:
        raise RunRecorderError(f"Run with id '{run_id}' was not found")

    run.output_answer = output_answer
    run.status = "completed"
    run.latency_ms = latency_ms
    run.prompt_tokens = prompt_tokens
    run.completion_tokens = completion_tokens
    run.estimated_cost = estimated_cost
    run.completed_at = utc_now()

    if metadata:
        run.metadata_json = {
            **(run.metadata_json or {}),
            **metadata
        }

    db.commit()
    db.refresh(run)

    logger.info("Completed run: %s", run.id)

    return RecordedRun(
        run_id=run.id,
        workflow_type=run.workflow_type,
        status=run.status
    )

def fail_run(
    db: Session,
    run_id: str,
    error_message: str,
    latency_ms: Optional[int] = None,
    metadata: Optional[dict[str, Any]] = None
) -> RecordedRun:
    run = db.get(Run, run_id)

    if run is None:
        raise RunRecorderError(f"Run with id '{run_id}' was not found")

    run.status = "failed"
    run.error_message = error_message
    run.latency_ms = latency_ms
    run.completed_at = utc_now()

    if metadata:
        run.metadata_json = {
            **(run.metadata_json or {}),
            **metadata
        }

    db.commit()
    db.refresh(run)

    logger.info("Failed run: %s", run.id)

    return RecordedRun(
        run_id=run.id,
        workflow_type=run.workflow_type,
        status=run.status
    )