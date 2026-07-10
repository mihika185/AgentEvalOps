import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ClipboardCheck,
  Clock,
  Gauge,
  Loader2,
  SearchCheck,
  ShieldCheck,
  X,
} from "lucide-react";

import { fetchJson } from "./api";
import type {
  AggregateQualityGateCheck,
  AggregateReport,
  ReportTarget,
} from "./types";

type ReportPanelProps = {
  target: ReportTarget;
  onClose: () => void;
};

export default function ReportPanel({ target, onClose }: ReportPanelProps) {
  const [report, setReport] = useState<AggregateReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadReport() {
    try {
      setLoading(true);
      setError(null);

      const path =
        target.type === "experiment"
          ? `/reports/experiments/${target.id}`
          : `/reports/benchmark-runs/${target.id}`;

      const data = await fetchJson<AggregateReport>(path);
      setReport(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load report");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadReport();
  }, [target.type, target.id]);

  return (
    <section className="inspection-panel report-panel">
      <div className="inspection-header">
        <div>
          <p className="eyebrow">
            {target.type === "experiment"
              ? "Experiment Report"
              : "Benchmark Report"}
          </p>
          <h2>{target.id}</h2>
        </div>

        <button className="ghost-button" onClick={onClose}>
          <X size={18} />
          Close
        </button>
      </div>

      {loading ? (
        <div className="inspection-state">
          <Loader2 className="spin" size={24} />
          <p>Loading aggregate report...</p>
        </div>
      ) : error ? (
        <div className="inspection-state error-card">
          <AlertTriangle size={24} />
          <p>{error}</p>
          <button onClick={loadReport}>Retry</button>
        </div>
      ) : report ? (
        <ReportContent report={report} />
      ) : null}
    </section>
  );
}

function ReportContent({ report }: { report: AggregateReport }) {
  const qualityGate = report.quality_gate_result;
  const readinessDecision = safeRecord(report.readiness_decision);
  const summary = safeRecord(report.summary);
  const details = safeRecord(report.details);

  const benchmarkRun = getRecord(summary, "benchmark_run");
  const cases = getRecord(summary, "cases");
  const failureCategories = getRecord(summary, "failure_categories");
  const benchmarkMetadata = getRecord(benchmarkRun, "metadata_json");
  const aggregateMetrics = qualityGate?.aggregate_metrics || {};

  const failedItems = getArray(details, "failed_items");
  const readinessStatus = readStatus(readinessDecision);
  const ready = readinessDecision.ready === true;

  const passRate =
    getNumber(cases, "pass_rate") ??
    getNumber(benchmarkRun, "pass_rate") ??
    aggregateMetrics.pass_rate ??
    qualityGate?.pass_rate;

  const failedCases =
    getNumber(cases, "failed_cases") ??
    getNumber(benchmarkRun, "failed_cases") ??
    aggregateMetrics.failed_cases;

  const totalCases =
    getNumber(cases, "total_cases") ??
    getNumber(benchmarkRun, "total_cases") ??
    aggregateMetrics.total_cases;

  const pipelineName = getString(benchmarkMetadata, "pipeline_config_name");
  const datasetName = getString(benchmarkMetadata, "dataset_name");

  return (
    <>
      <div className="report-status-grid">
        <StatusCard
          title="Readiness"
          value={readinessStatus}
          passed={ready}
          icon={<ClipboardCheck size={19} />}
        />
        <StatusCard
          title="Aggregate Gates"
          value={qualityGate?.overall_passed ? "Passed" : "Needs review"}
          passed={qualityGate?.overall_passed === true}
          icon={<ShieldCheck size={19} />}
        />
        <StatusCard
          title="Case Pass Rate"
          value={formatPercent(passRate)}
          passed={(passRate || 0) >= 0.8}
          icon={<CheckCircle2 size={19} />}
        />
        <StatusCard
          title="Failed Cases"
          value={formatNumber(failedCases)}
          passed={(failedCases || 0) === 0}
          icon={<AlertTriangle size={19} />}
        />
      </div>

      <div className={ready ? "report-callout report-ready" : "report-callout report-needs-review"}>
        <div>
          <strong>{ready ? "Ready for demo/release" : "Needs attention"}</strong>
          <p>{getString(readinessDecision, "recommendation") || "No recommendation returned."}</p>
        </div>
        <span>{formatUnknown(readinessDecision.reasons)}</span>
      </div>

      {pipelineName || datasetName ? (
        <DetailCard title="Report Context">
          <div className="report-context-grid">
            <InfoRow label="Dataset" value={datasetName || "—"} />
            <InfoRow label="Pipeline" value={pipelineName || "—"} />
            <InfoRow
              label="Retrieval"
              value={getString(benchmarkMetadata, "retrieval_provider") || "—"}
            />
            <InfoRow
              label="Rerank"
              value={formatUnknown(benchmarkMetadata.rerank)}
            />
            <InfoRow
              label="Quality Profile"
              value={getString(benchmarkMetadata, "quality_gate_profile") || "—"}
            />
            <InfoRow
              label="Generated At"
              value={report.generated_at || "—"}
            />
          </div>
        </DetailCard>
      ) : null}

      <MetricSection
        title="Pass / Fail Summary"
        icon={<BarChart3 size={18} />}
        metrics={[
          ["Total Cases", formatNumber(totalCases), "Benchmark cases evaluated"],
          ["Passed Cases", formatNumber(getNumber(cases, "passed_cases") ?? aggregateMetrics.passed_cases), "Cases that passed"],
          ["Failed Cases", formatNumber(failedCases), "Cases that failed"],
          ["Pass Rate", formatPercent(passRate), "Overall benchmark pass rate"],
          ["Answerable Accuracy", formatPercent(aggregateMetrics.answerable_accuracy), "Answerable case success"],
          ["Failed Case Rate", formatPercent(getNumber(cases, "failed_case_rate") ?? aggregateMetrics.failed_case_rate), "Lower is better"],
        ]}
      />

      <MetricSection
        title="Retrieval Metrics"
        icon={<SearchCheck size={18} />}
        metrics={[
          ["Recall@k", formatScore(getNumber(benchmarkMetadata, "average_recall_at_k")), "Relevant chunks retrieved"],
          ["Precision@k", formatScore(getNumber(benchmarkMetadata, "average_precision_at_k")), "Relevant chunk density"],
          ["MRR", formatScore(getNumber(benchmarkMetadata, "average_mrr")), "First relevant rank quality"],
          ["nDCG@k", formatScore(getNumber(benchmarkMetadata, "average_ndcg_at_k")), "Ranking quality"],
        ]}
      />

      <MetricSection
        title="Answer Quality"
        icon={<Gauge size={18} />}
        metrics={[
          ["Overall Quality", formatScore(aggregateMetrics.average_overall_quality_score), "Combined quality score"],
          ["Answer Support", formatScore(aggregateMetrics.average_answer_support_score), "Grounded in retrieved context"],
          ["Query Relevance", formatScore(aggregateMetrics.average_query_answer_relevance_score), "Directly answers query"],
          ["Hallucination Risk", formatScore(aggregateMetrics.average_hallucination_risk), "Lower is better"],
        ]}
      />

      <MetricSection
        title="Latency, Cost, and Tokens"
        icon={<Clock size={18} />}
        metrics={[
          ["Average Latency", formatMs(aggregateMetrics.average_latency_ms), "Average run latency"],
          ["Prompt Tokens", formatNumber(aggregateMetrics.average_prompt_tokens), "Average prompt tokens"],
          ["Completion Tokens", formatNumber(aggregateMetrics.average_completion_tokens), "Average completion tokens"],
          ["Total Tokens", formatNumber(aggregateMetrics.average_total_tokens), "Average total tokens"],
          ["Average Cost", formatCurrency(aggregateMetrics.average_estimated_cost), "Estimated average cost"],
          ["Total Cost", formatCurrency(aggregateMetrics.total_estimated_cost), "Estimated total cost"],
        ]}
      />

      {qualityGate ? (
        <DetailCard title={`Quality Gate Checks (${qualityGate.checks.length})`}>
          <div className="quality-gate-summary">
            <span className={qualityGate.overall_passed ? "badge success" : "badge danger"}>
              {qualityGate.overall_passed ? "all gates passed" : "gates failed"}
            </span>
            <span>
              {qualityGate.passed_count}/{qualityGate.total_gates} checks passed
            </span>
            <span>{formatPercent(qualityGate.pass_rate)} gate pass rate</span>
          </div>

          <div className="check-list">
            {qualityGate.checks.map((check) => (
              <GateCheckCard key={check.gate_id} check={check} />
            ))}
          </div>
        </DetailCard>
      ) : (
        <DetailCard title="Quality Gate Checks">
          <p className="muted-text">No aggregate quality gate result found.</p>
        </DetailCard>
      )}

      <DetailCard title={`Failed Items (${failedItems.length})`}>
        {failedItems.length > 0 ? (
          <div className="failed-item-list">
            {failedItems.map((item, index) => (
              <FailedItemCard key={index} item={safeRecord(item)} />
            ))}
          </div>
        ) : (
          <div className="empty-success">
            <CheckCircle2 size={18} />
            <span>No failed items. This report is clean.</span>
          </div>
        )}
      </DetailCard>

      {Object.keys(failureCategories).length > 0 ? (
        <DetailCard title="Failure Categories">
          <div className="report-metric-grid">
            {Object.entries(failureCategories).map(([key, value]) => (
              <MetricTile
                key={key}
                label={toLabel(key)}
                value={formatUnknown(value)}
                hint="Failure category"
              />
            ))}
          </div>
        </DetailCard>
      ) : null}

      <details className="report-details-toggle">
        <summary>Raw report details</summary>
        <JsonBlock value={details} />
      </details>
    </>
  );
}

function MetricSection({
  title,
  icon,
  metrics,
}: {
  title: string;
  icon: ReactNode;
  metrics: Array<[string, string, string]>;
}) {
  return (
    <DetailCard title={title}>
      <div className="report-section-heading">
        {icon}
        <span>{title}</span>
      </div>

      <div className="report-metric-grid">
        {metrics.map(([label, value, hint]) => (
          <MetricTile key={label} label={label} value={value} hint={hint} />
        ))}
      </div>
    </DetailCard>
  );
}

function MetricTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="report-metric-card">
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{hint}</span>
    </div>
  );
}

function StatusCard({
  title,
  value,
  passed,
  icon,
}: {
  title: string;
  value: string;
  passed: boolean;
  icon: ReactNode;
}) {
  return (
    <div className={passed ? "status-card status-pass" : "status-card status-warn"}>
      <div>
        {icon}
        <span>{title}</span>
      </div>
      <strong>{value}</strong>
    </div>
  );
}

function DetailCard({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="detail-card">
      <div className="detail-title">
        <h3>{title}</h3>
      </div>
      {children}
    </div>
  );
}

function InfoRow({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="info-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function GateCheckCard({ check }: { check: AggregateQualityGateCheck }) {
  return (
    <div className={check.passed ? "check-card check-pass" : "check-card check-fail"}>
      <div>
        <strong>{check.gate_name}</strong>
        <p>
          {toLabel(check.metric_name)} {check.operator}{" "}
          {formatMetric(check.threshold)}
        </p>
      </div>

      <div className="check-card-meta">
        <span className={check.passed ? "badge success" : "badge danger"}>
          {check.passed ? "passed" : "failed"}
        </span>
        <span>{formatMetric(check.metric_value)}</span>
      </div>

      {check.failure_reason ? (
        <p className="check-failure">{check.failure_reason}</p>
      ) : null}
    </div>
  );
}

function FailedItemCard({ item }: { item: Record<string, unknown> }) {
  const metrics = getRecord(item, "metrics_json");

  return (
    <div className="failed-item-card">
      <div className="failed-item-top">
        <div>
          <strong>{getString(item, "question") || "Failed benchmark item"}</strong>
          <p>{getString(item, "failure_reason") || "No failure reason provided."}</p>
        </div>

        <span className="badge danger">failed</span>
      </div>

      <div className="failed-answer-box">
        <span>Actual answer</span>
        <p>{getString(item, "actual_answer") || "—"}</p>
      </div>

      <div className="failed-item-metrics">
        <InfoRow
          label="Expected Behavior"
          value={getString(item, "expected_behavior") || "—"}
        />
        <InfoRow
          label="Quality Gate Passed"
          value={formatUnknown(item.quality_gate_passed)}
        />
        <InfoRow
          label="Response Blocked"
          value={formatUnknown(item.response_blocked_by_quality_gate)}
        />
        <InfoRow
          label="Overall Quality"
          value={formatScore(getNumber(metrics, "overall_quality_score"))}
        />
        <InfoRow
          label="Hallucination Risk"
          value={formatScore(getNumber(metrics, "hallucination_risk"))}
        />
        <InfoRow
          label="Latency"
          value={formatMs(getNumber(item, "latency_ms"))}
        />
      </div>
    </div>
  );
}

function JsonBlock({ value }: { value: Record<string, unknown> }) {
  return (
    <div className="json-block json-block-full">
      <pre>{JSON.stringify(value || {}, null, 2)}</pre>
    </div>
  );
}

function readStatus(readinessDecision: Record<string, unknown>) {
  const status = readinessDecision.status;

  if (typeof status === "string") {
    return status;
  }

  const ready = readinessDecision.ready;

  if (ready === true) {
    return "ready";
  }

  if (ready === false) {
    return "not ready";
  }

  return "unknown";
}

function toLabel(value: string) {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatUnknown(value: unknown): string {
  if (value === null || value === undefined) {
    return "—";
  }

  if (typeof value === "number") {
    return formatMetric(value);
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  if (typeof value === "string") {
    return value;
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return "None";
    }

    if (value.every((item) => typeof item !== "object")) {
      return value.join(", ");
    }

    return `${value.length} items`;
  }

  if (isRecord(value)) {
    return summarizeObject(value);
  }

  return String(value);
}

function summarizeObject(value: Record<string, unknown>): string {
  if (typeof value.name === "string" && typeof value.id === "string") {
    return `${value.name} (${value.id})`;
  }

  if (typeof value.name === "string") {
    return value.name;
  }

  if (typeof value.id === "string") {
    return value.id;
  }

  if ("total_runs" in value && "completed_runs" in value) {
    return `${formatUnknown(value.completed_runs)}/${formatUnknown(
      value.total_runs
    )} runs completed`;
  }

  if ("total_cases" in value && "passed_cases" in value) {
    return `${formatUnknown(value.passed_cases)}/${formatUnknown(
      value.total_cases
    )} cases passed`;
  }

  if ("average_overall_quality_score" in value) {
    return `Average quality ${formatUnknown(
      value.average_overall_quality_score
    )}`;
  }

  return `${Object.keys(value).length} fields`;
}

function safeRecord(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

function getRecord(
  value: Record<string, unknown>,
  key: string
): Record<string, unknown> {
  const nested = value[key];

  return isRecord(nested) ? nested : {};
}

function getArray(value: Record<string, unknown>, key: string): unknown[] {
  const nested = value[key];

  return Array.isArray(nested) ? nested : [];
}

function getString(value: Record<string, unknown>, key: string): string | null {
  const nested = value[key];

  return typeof nested === "string" ? nested : null;
}

function getNumber(value: Record<string, unknown>, key: string): number | null {
  const nested = value[key];

  return typeof nested === "number" ? nested : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatMetric(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  if (Number.isInteger(value)) {
    return value.toString();
  }

  return Number(value).toFixed(4);
}

function formatNumber(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: 2,
  });
}

function formatScore(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return Number(value).toFixed(4);
}

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatMs(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `${Number(value).toFixed(1)} ms`;
}

function formatCurrency(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `$${Number(value).toFixed(4)}`;
}