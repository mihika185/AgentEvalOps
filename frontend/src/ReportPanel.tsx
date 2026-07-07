import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ClipboardCheck,
  Loader2,
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
  const readinessStatus = readStatus(report.readiness_decision);

  return (
    <>
      <div className="report-status-grid">
        <StatusCard
          title="Readiness"
          value={readinessStatus}
          passed={readinessStatus.toLowerCase() === "ready"}
          icon={<ClipboardCheck size={19} />}
        />
        <StatusCard
          title="Aggregate Gates"
          value={qualityGate?.overall_passed ? "Passed" : "Needs review"}
          passed={qualityGate?.overall_passed === true}
          icon={<ShieldCheck size={19} />}
        />
        <StatusCard
          title="Gate Pass Rate"
          value={formatPercent(qualityGate?.pass_rate)}
          passed={(qualityGate?.pass_rate || 0) >= 0.8}
          icon={<CheckCircle2 size={19} />}
        />
      </div>

      <div className="inspection-grid">
        <DetailCard title="Summary">
          {Object.entries(report.summary || {}).map(([key, value]) => (
            <InfoRow key={key} label={toLabel(key)} value={formatUnknown(value)} />
          ))}
        </DetailCard>

        <DetailCard title="Readiness Decision">
          {Object.entries(report.readiness_decision || {}).map(([key, value]) => (
            <InfoRow key={key} label={toLabel(key)} value={formatUnknown(value)} />
          ))}
        </DetailCard>
      </div>

      {qualityGate ? (
        <>
          <DetailCard title="Aggregate Metrics">
            <div className="metric-result-grid">
              {Object.entries(qualityGate.aggregate_metrics || {}).map(
                ([key, value]) => (
                  <div className="metric-result-card" key={key}>
                    <p>{toLabel(key)}</p>
                    <strong>{formatMetric(value)}</strong>
                    <span>aggregate metric</span>
                  </div>
                )
              )}
            </div>
          </DetailCard>

          <DetailCard title={`Quality Gate Checks (${qualityGate.checks.length})`}>
            <div className="check-list">
              {qualityGate.checks.map((check) => (
                <GateCheckCard key={check.gate_id} check={check} />
              ))}
            </div>
          </DetailCard>
        </>
      ) : (
        <DetailCard title="Quality Gate Checks">
          <p className="muted-text">No aggregate quality gate result found.</p>
        </DetailCard>
      )}

      <DetailCard title="Report Details">
        <JsonBlock value={report.details} />
      </DetailCard>
    </>
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

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `${(Number(value) * 100).toFixed(1)}%`;
}