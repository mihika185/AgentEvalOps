import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Loader2,
  Network,
  ShieldCheck,
  X,
} from "lucide-react";

import { fetchJson } from "./api";
import type {
  EvaluationResult,
  RunInspection,
  TraceStep,
} from "./types";

type RunInspectionPanelProps = {
  runId: string;
  onClose: () => void;
};

export default function RunInspectionPanel({
  runId,
  onClose,
}: RunInspectionPanelProps) {
  const [inspection, setInspection] = useState<RunInspection | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadInspection() {
    try {
      setLoading(true);
      setError(null);

      const data = await fetchJson<RunInspection>(
        `/runs/${runId}/inspection`
      );

      setInspection(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load run inspection"
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadInspection();
  }, [runId]);

  return (
    <section className="inspection-panel">
      <div className="inspection-header">
        <div>
          <p className="eyebrow">Run Inspection</p>
          <h2>{runId}</h2>
        </div>

        <button className="ghost-button" onClick={onClose}>
          <X size={18} />
          Close
        </button>
      </div>

      {loading ? (
        <div className="inspection-state">
          <Loader2 className="spin" size={24} />
          <p>Loading run trace and evaluations...</p>
        </div>
      ) : error ? (
        <div className="inspection-state error-card">
          <AlertTriangle size={24} />
          <p>{error}</p>
          <button onClick={loadInspection}>Retry</button>
        </div>
      ) : inspection ? (
        <RunInspectionContent inspection={inspection} />
      ) : null}
    </section>
  );
}

function RunInspectionContent({
  inspection,
}: {
  inspection: RunInspection;
}) {
  const answer = inspection.run.output_answer || "No answer stored for this run.";

  return (
    <>
      <div className="inspection-grid">
        <DetailCard title="Run Summary">
          <InfoRow label="Workflow" value={inspection.run.workflow_type} />
          <InfoRow label="Status" value={inspection.run.status} />
          <InfoRow label="Latency" value={formatMs(inspection.run.latency_ms)} />
          <InfoRow label="Created" value={formatDate(inspection.run.created_at)} />
          <InfoRow
            label="Completed"
            value={formatDate(inspection.run.completed_at)}
          />
        </DetailCard>

        <DetailCard title="Inspection Summary">
          {Object.entries(inspection.summary).map(([key, value]) => (
            <InfoRow
              key={key}
              label={toLabel(key)}
              value={formatUnknown(value)}
            />
          ))}
        </DetailCard>
      </div>

      <DetailCard title="Question">
        <p className="answer-text">{inspection.run.input_query}</p>
      </DetailCard>

      <DetailCard title="Final Answer">
        <p className="answer-text">{answer}</p>
      </DetailCard>

      <DetailCard
        title={`Trace Steps (${inspection.trace_steps.length})`}
        icon={<Network size={18} />}
      >
        <div className="trace-list">
          {inspection.trace_steps.map((step, index) => (
            <TraceStepCard
              key={`${step.step_index}-${step.step_name || index}`}
              step={step}
            />
          ))}
        </div>
      </DetailCard>

      <DetailCard
        title={`Quality Gates (${inspection.quality_gate_results.length})`}
        icon={<ShieldCheck size={18} />}
      >
        <MetricGrid results={inspection.quality_gate_results} />
      </DetailCard>

      <DetailCard
        title={`Evaluation Results (${inspection.evaluation_results.length})`}
        icon={<CheckCircle2 size={18} />}
      >
        <MetricGrid results={inspection.evaluation_results} />
      </DetailCard>
    </>
  );
}

function DetailCard({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="detail-card">
      <div className="detail-title">
        {icon}
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

function TraceStepCard({ step }: { step: TraceStep }) {
  return (
    <div className="trace-step-card">
      <div className="trace-step-top">
        <div>
          <strong>
            #{step.step_index} {step.step_name || "Unnamed step"}
          </strong>
          <p>{step.step_type || "trace step"}</p>
        </div>

        <div className="trace-step-meta">
          <span className="badge neutral">{step.status || "recorded"}</span>
          <span>
            <Clock size={14} />
            {formatMs(step.latency_ms)}
          </span>
        </div>
      </div>

      {step.error_message ? (
        <div className="trace-error">
          <AlertTriangle size={15} />
          {step.error_message}
        </div>
      ) : null}

      <div className="json-grid">
        <JsonBlock title="Input" value={step.input_data} />
        <JsonBlock title="Output" value={step.output_data} />
        <JsonBlock title="Metadata" value={step.metadata_json} />
      </div>
    </div>
  );
}

function MetricGrid({ results }: { results: EvaluationResult[] }) {
  if (results.length === 0) {
    return <p className="muted-text">No results recorded.</p>;
  }

  return (
    <div className="metric-result-grid">
      {results.map((result, index) => (
        <div
          className="metric-result-card"
          key={`${result.evaluator_type}-${result.metric_name}-${index}`}
        >
          <p>{toLabel(result.metric_name)}</p>
          <strong>{formatMetric(result.metric_value)}</strong>
          <span>{result.evaluator_type}</span>
        </div>
      ))}
    </div>
  );
}

function JsonBlock({
  title,
  value,
}: {
  title: string;
  value: Record<string, unknown> | null | undefined;
}) {
  if (!value || Object.keys(value).length === 0) {
    return (
      <div className="json-block">
        <p>{title}</p>
        <pre>Empty</pre>
      </div>
    );
  }

  return (
    <div className="json-block">
      <p>{title}</p>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </div>
  );
}

function toLabel(value: string) {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatMs(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `${Math.round(Number(value))} ms`;
}

function formatMetric(value: number) {
  if (Number.isInteger(value)) {
    return value.toString();
  }

  return Number(value).toFixed(4);
}

function formatDate(value: string | null | undefined) {
  if (!value) {
    return "—";
  }

  return new Date(value).toLocaleString();
}

function formatUnknown(value: unknown) {
  if (value === null || value === undefined) {
    return "—";
  }

  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(4);
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  if (Array.isArray(value)) {
    return value.join(", ");
  }

  if (typeof value === "object") {
    return JSON.stringify(value);
  }

  return String(value);
}