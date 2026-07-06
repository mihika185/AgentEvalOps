from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.database.models import (
    BenchmarkDataset,
    BenchmarkRun,
    BenchmarkRunItem,
    EvaluationResult,
    Experiment,
    Run,
)


IMPORTANT_EXPERIMENT_METRICS = {
    "overall_quality_score",
    "overall_agent_score",
    "answer_support_score",
    "query_answer_relevance_score",
    "hallucination_risk",
    "quality_gate_pass_rate",
    "quality_gate_overall_pass",
    "tool_selection_accuracy",
    "answer_correctness",
}

def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a markdown benchmark report for AgentEvalOps."
    )

    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Optional benchmark dataset id to filter benchmark runs.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional experiment id to include experiment-level summary.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of recent benchmark runs to include.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_report.md",
        help="Path where the markdown report should be written.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    db_generator = get_db()
    db = next(db_generator)

    try:
        report = generate_benchmark_report(
            db=db,
            dataset_id=args.dataset_id,
            experiment_id=args.experiment_id,
            limit=args.limit,
        )

        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")

        print(f"Wrote benchmark report to {output_path}")

    finally:
        try:
            next(db_generator)
        except StopIteration:
            pass

def generate_benchmark_report(
    db: Session,
    dataset_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    limit: int = 10,
)->str:
    benchmark_runs = load_benchmark_runs(
        db=db,
        dataset_id=dataset_id,
        limit=limit,
    )

    lines = [
        "# AgentEvalOps Benchmark Report",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Scope",
        "",
        f"- Dataset filter: `{dataset_id or 'latest benchmark runs'}`",
        f"- Experiment filter: `{experiment_id or 'not included'}`",
        f"- Benchmark runs included: `{len(benchmark_runs)}`",
        "",
    ]

    if experiment_id:
        experiment = db.get(Experiment, experiment_id)

        if experiment is None:
            raise ValueError(f"Experiment was not found: {experiment_id}")

        experiment_runs = load_experiment_runs(db, experiment_id)
        experiment_evaluations = load_evaluation_results_for_runs(
            db=db,
            run_ids=[run.id for run in experiment_runs],
        )

        lines.extend(
            build_experiment_report_section(
                experiment=experiment,
                runs=experiment_runs,
                evaluation_results=experiment_evaluations,
            )
        )

    if not benchmark_runs:
        lines.extend(
            [
                "## Benchmark Runs",
                "",
                "No benchmark runs were found for this scope.",
                "",
            ]
        )

        return "\n".join(lines).rstrip() + "\n"

    datasets = load_datasets_for_benchmark_runs(db, benchmark_runs)
    best_run = select_best_benchmark_run(benchmark_runs)

    lines.extend(build_benchmark_summary_section(benchmark_runs))
    lines.extend(build_best_benchmark_run_section(best_run, datasets))
    lines.extend(build_benchmark_run_details_section(db, benchmark_runs, datasets))

    return "\n".join(lines).rstrip() + "\n"

def load_benchmark_runs(
    db: Session,
    dataset_id: Optional[str],
    limit: int,
)->list[BenchmarkRun]:
    statement = select(BenchmarkRun)

    if dataset_id:
        statement = statement.where(BenchmarkRun.dataset_id == dataset_id)

    statement = (
        statement
        .order_by(BenchmarkRun.started_at.desc())
        .limit(limit)
    )

    return list(db.execute(statement).scalars().all())

def load_benchmark_items(
    db: Session,
    benchmark_run_id: str,
)->list[BenchmarkRunItem]:
    statement = (
        select(BenchmarkRunItem)
        .where(BenchmarkRunItem.benchmark_run_id == benchmark_run_id)
        .order_by(BenchmarkRunItem.created_at.asc())
    )

    return list(db.execute(statement).scalars().all())

def load_datasets_for_benchmark_runs(
    db: Session,
    benchmark_runs: list[BenchmarkRun],
)->dict[str, BenchmarkDataset]:
    dataset_ids = sorted(
        {
            run.dataset_id
            for run in benchmark_runs
            if run.dataset_id
        }
    )

    if not dataset_ids:
        return {}

    statement = select(BenchmarkDataset).where(BenchmarkDataset.id.in_(dataset_ids))

    datasets = db.execute(statement).scalars().all()

    return {
        dataset.id: dataset
        for dataset in datasets
    }

def load_experiment_runs(
    db: Session,
    experiment_id: str,
) ->list[Run]:
    statement = (
        select(Run)
        .where(Run.experiment_id == experiment_id)
        .order_by(Run.created_at.desc())
    )

    return list(db.execute(statement).scalars().all())

def load_evaluation_results_for_runs(
    db: Session,
    run_ids: list[str],
) ->list[EvaluationResult]:
    if not run_ids:
        return []

    statement = select(EvaluationResult).where(EvaluationResult.run_id.in_(run_ids))

    return list(db.execute(statement).scalars().all())

def build_experiment_report_section(
    experiment: Experiment,
    runs: list[Run],
    evaluation_results: list[EvaluationResult],
)-> list[str]:
    workflow_counts = count_by_attribute(runs, "workflow_type")
    status_counts = count_by_attribute(runs, "status")
    metric_summaries = summarize_metric_values(evaluation_results)

    completed_runs = status_counts.get("completed", 0)
    failed_runs = status_counts.get("failed", 0)

    lines = [
        "## Experiment Summary",
        "",
        f"- Experiment ID: `{experiment.id}`",
        f"- Name: {safe_markdown_text(experiment.name)}",
        f"- Total runs: `{len(runs)}`",
        f"- Completed runs: `{completed_runs}`",
        f"- Failed runs: `{failed_runs}`",
        f"- Average latency: `{format_value(average_latency([run.latency_ms for run in runs]))} ms`",
        "",
        "### Runs by Workflow",
        "",
        "| Workflow Type | Count |",
        "|---|---:|",
    ]

    for workflow_type, count in workflow_counts.items():
        lines.append(f"| `{workflow_type}` | {count} |")

    lines.extend(
        [
            "",
            "### Runs by Status",
            "",
            "| Status | Count |",
            "|---|---:|",
        ]
    )

    for status, count in status_counts.items():
        lines.append(f"| `{status}` | {count} |")

    important_summaries = [
        summary
        for summary in metric_summaries
        if summary["metric_name"] in IMPORTANT_EXPERIMENT_METRICS
    ]

    if important_summaries:
        lines.extend(
            [
                "",
                "### Experiment Metric Highlights",
                "",
                "| Evaluator | Metric | Count | Average | Min | Max |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )

        for summary in important_summaries:
            lines.append(
                "| "
                f"`{summary['evaluator_type']}` | "
                f"`{summary['metric_name']}` | "
                f"{summary['count']} | "
                f"{format_value(summary['average_value'])} | "
                f"{format_value(summary['min_value'])} | "
                f"{format_value(summary['max_value'])} |"
            )

    lines.append("")

    return lines

def build_benchmark_summary_section(
    benchmark_runs: list[BenchmarkRun],
)->list[str]:
    lines = [
        "## Benchmark Run Summary",
        "",
        "| Benchmark Run | Pipeline | Status | Cases | Pass Rate | Avg Quality | Avg Latency | Recall@k | Precision@k | MRR | nDCG@k |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for run in benchmark_runs:
        metadata = ensure_dict(run.metadata_json)

        lines.append(
            "| "
            f"`{run.id}` | "
            f"{safe_markdown_text(pipeline_name(run))} | "
            f"`{run.status}` | "
            f"{run.passed_cases}/{run.total_cases} | "
            f"{format_value(pass_rate(run))} | "
            f"{format_value(run.average_overall_quality_score)} | "
            f"{format_value(run.average_latency_ms)} | "
            f"{format_value(metadata.get('average_recall_at_k'))} | "
            f"{format_value(metadata.get('average_precision_at_k'))} | "
            f"{format_value(metadata.get('average_mrr'))} | "
            f"{format_value(metadata.get('average_ndcg_at_k'))} |"
        )

    lines.append("")

    return lines

def build_best_benchmark_run_section(
    best_run: Optional[BenchmarkRun],
    datasets: dict[str, BenchmarkDataset],
)->list[str]:
    lines = [
        "## Best Benchmark Run",
        "",
    ]

    if best_run is None:
        lines.extend(
            [
                "No completed benchmark run was available for ranking.",
                "",
            ]
        )

        return lines

    dataset = datasets.get(best_run.dataset_id)
    metadata = ensure_dict(best_run.metadata_json)

    lines.extend(
        [
            f"- Benchmark run: `{best_run.id}`",
            f"- Dataset: `{dataset.name if dataset else best_run.dataset_id}`",
            f"- Pipeline: {safe_markdown_text(pipeline_name(best_run))}",
            f"- Pass rate: `{format_value(pass_rate(best_run))}`",
            f"- Average quality: `{format_value(best_run.average_overall_quality_score)}`",
            f"- Average latency: `{format_value(best_run.average_latency_ms)} ms`",
            f"- Average recall@k: `{format_value(metadata.get('average_recall_at_k'))}`",
            f"- Average MRR: `{format_value(metadata.get('average_mrr'))}`",
            f"- Average nDCG@k: `{format_value(metadata.get('average_ndcg_at_k'))}`",
            "",
            "Selection rule: highest pass rate, then highest average overall quality.",
            "",
        ]
    )

    return lines

def build_benchmark_run_details_section(
    db: Session,
    benchmark_runs: list[BenchmarkRun],
    datasets: dict[str, BenchmarkDataset],
) ->list[str]:
    lines = [
        "## Benchmark Run Details",
        "",
    ]

    for run in benchmark_runs:
        dataset = datasets.get(run.dataset_id)
        items = load_benchmark_items(db, run.id)

        lines.extend(
            [
                f"### `{run.id}` — {safe_markdown_text(pipeline_name(run))}",
                "",
                f"- Dataset: `{dataset.name if dataset else run.dataset_id}`",
                f"- Status: `{run.status}`",
                f"- Cases: `{run.passed_cases}/{run.total_cases}`",
                f"- Pass rate: `{format_value(pass_rate(run))}`",
                f"- Average overall quality: `{format_value(run.average_overall_quality_score)}`",
                "",
                "| Result | Quality | Recall | MRR | Latency | Question |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )

        for item in items:
            metrics = ensure_dict(item.metrics_json)
            result_label = "PASS" if item.passed else "FAIL"

            lines.append(
                "| "
                f"`{result_label}` | "
                f"{format_value(metrics.get('overall_quality_score'))} | "
                f"{format_value(metric_by_prefix(metrics, 'recall_at_'))} | "
                f"{format_value(metrics.get('mrr'))} | "
                f"{format_value(item.latency_ms)} | "
                f"{safe_markdown_text(truncate(item.question, 100))} |"
            )

        failed_items = [
            item
            for item in items
            if not item.passed
        ]

        if failed_items:
            lines.extend(
                [
                    "",
                    "**Failures**",
                    "",
                ]
            )

            for item in failed_items:
                lines.append(
                    f"- `{item.id}`: {safe_markdown_text(item.failure_reason or 'Unknown failure')}"
                )

        lines.append("")

    return lines

def select_best_benchmark_run(
    benchmark_runs: list[BenchmarkRun],
) -> Optional[BenchmarkRun]:
    completed_runs = [
        run
        for run in benchmark_runs
        if run.status == "completed"
    ]

    if not completed_runs:
        return None

    return max(
        completed_runs,
        key=lambda run: (
            pass_rate(run) or 0.0,
            run.average_overall_quality_score or 0.0,
        ),
    )

def pipeline_name(run: BenchmarkRun) -> str:
    metadata = ensure_dict(run.metadata_json)

    return (
        metadata.get("pipeline_config_name")
        or metadata.get("pipeline_config_id")
        or "Direct benchmark run"
    )

def pass_rate(run: BenchmarkRun) -> Optional[float]:
    if not run.total_cases:
        return None

    return round(run.passed_cases / run.total_cases, 4)

def count_by_attribute(items: list[Any], attribute_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}

    for item in items:
        raw_value = getattr(item, attribute_name, None)
        value = str(raw_value).strip() if raw_value is not None else "unknown"

        if not value:
            value = "unknown"

        counts[value] = counts.get(value, 0) + 1

    return dict(sorted(counts.items()))

def summarize_metric_values(
    evaluation_results: list[EvaluationResult],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}

    for result in evaluation_results:
        key = (
            result.evaluator_type,
            result.metric_name,
        )

        grouped.setdefault(key, []).append(float(result.metric_value))

    summaries = []

    for key, values in grouped.items():
        evaluator_type, metric_name = key

        summaries.append(
            {
                "evaluator_type": evaluator_type,
                "metric_name": metric_name,
                "count": len(values),
                "average_value": round(sum(values) / len(values), 4),
                "min_value": round(min(values), 4),
                "max_value": round(max(values), 4),
            }
        )

    return sorted(
        summaries,
        key=lambda summary: (
            summary["evaluator_type"],
            summary["metric_name"],
        ),
    )

def average_latency(values: list[Optional[int]]) -> Optional[float]:
    cleaned_values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not cleaned_values:
        return None

    return round(sum(cleaned_values) / len(cleaned_values), 2)

def metric_by_prefix(metrics: dict[str, Any], prefix: str) -> Optional[float]:
    for key, value in metrics.items():
        if key.startswith(prefix):
            return float(value)

    return None

def ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    return {}

def format_value(value: Any) -> str:
    if value is None:
        return "-"

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        rounded = round(value, 4)
        text = f"{rounded:.4f}".rstrip("0").rstrip(".")

        return text if text else "0"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    rounded = round(numeric_value, 4)
    return f"{rounded:.4f}".rstrip("0").rstrip(".")

def truncate(text: Optional[str], max_length: int) -> str:
    if text is None:
        return ""

    cleaned = " ".join(str(text).split())

    if len(cleaned) <= max_length:
        return cleaned

    return cleaned[: max_length - 3].rstrip() + "..."

def safe_markdown_text(value: Optional[str]) -> str:
    if value is None:
        return ""

    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()