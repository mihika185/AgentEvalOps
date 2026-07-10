import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Plus,
  RefreshCw,
  ShieldCheck,
  Trash2,
} from "lucide-react";

import {
  deleteJson,
  fetchJson,
  patchJson,
  postJson,
} from "./api";

type GateOperator = ">=" | "<=" | ">" | "<" | "==";

type QualityGate = {
  id: string;
  name: string;
  metric_name: string;
  operator: GateOperator;
  threshold: number;
  is_active: boolean;
  metadata_json: Record<string, unknown>;
  created_at: string;
};

type QualityGateCreatePayload = {
  name: string;
  metric_name: string;
  operator: GateOperator;
  threshold: number;
  is_active: boolean;
  metadata_json: Record<string, unknown>;
};

type QualityGateCheckResponse = {
  gate_status: "passed" | "failed";
  total_checks: number;
  passed_checks: number;
  failed_checks: number;
  checks: Array<{
    gate_id: string;
    name: string;
    metric_name: string;
    operator: string;
    threshold: number;
    actual_value: number | null;
    passed: boolean;
    reason: string;
  }>;
};

const DEFAULT_FORM: QualityGateCreatePayload = {
  name: "Minimum Overall Quality",
  metric_name: "overall_quality_score",
  operator: ">=",
  threshold: 0.7,
  is_active: true,
  metadata_json: {},
};

const SAMPLE_METRICS = {
  answer_support_score: 0.92,
  query_answer_relevance_score: 0.88,
  hallucination_risk: 0.04,
  top_retrieval_score: 0.81,
  overall_quality_score: 0.9,
  source_chunk_count: 2,
  pass_rate: 0.95,
  failed_case_rate: 0.05,
  average_latency_ms: 1200,
  average_estimated_cost: 0.01,
};

export default function QualityGateManager() {
  const [gates, setGates] = useState<QualityGate[]>([]);
  const [form, setForm] = useState<QualityGateCreatePayload>(DEFAULT_FORM);
  const [checkResult, setCheckResult] = useState<QualityGateCheckResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadGates() {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchJson<QualityGate[]>("/quality-gates");
      setGates(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load quality gates");
    } finally {
      setLoading(false);
    }
  }

  async function seedDefaults() {
    try {
      setWorking(true);
      setError(null);
      await postJson<{ created_count: number }, Record<string, never>>(
        "/quality-gates/defaults",
        {}
      );
      await loadGates();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to seed default gates");
    } finally {
      setWorking(false);
    }
  }

  async function createGate() {
    if (!form.name.trim() || !form.metric_name.trim()) {
      setError("Gate name and metric name are required.");
      return;
    }

    try {
      setWorking(true);
      setError(null);

      await postJson<QualityGate, QualityGateCreatePayload>("/quality-gates", {
        ...form,
        name: form.name.trim(),
        metric_name: form.metric_name.trim(),
        threshold: Number(form.threshold),
      });

      setForm(DEFAULT_FORM);
      await loadGates();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create quality gate");
    } finally {
      setWorking(false);
    }
  }

  async function toggleGate(gate: QualityGate) {
    try {
      setWorking(true);
      setError(null);

      await patchJson<QualityGate, { is_active: boolean }>(
        `/quality-gates/${gate.id}`,
        {
          is_active: !gate.is_active,
        }
      );

      await loadGates();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update quality gate");
    } finally {
      setWorking(false);
    }
  }

  async function deleteGate(gate: QualityGate) {
    try {
      setWorking(true);
      setError(null);

      await deleteJson(`/quality-gates/${gate.id}`);
      await loadGates();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete quality gate");
    } finally {
      setWorking(false);
    }
  }

  async function runSampleCheck() {
    try {
      setWorking(true);
      setError(null);

      const data = await postJson<
        QualityGateCheckResponse,
        {
          metrics: Record<string, number>;
          active_only: boolean;
          fail_on_missing_metrics: boolean;
        }
      >("/quality-gates/check", {
        metrics: SAMPLE_METRICS,
        active_only: true,
        fail_on_missing_metrics: false,
      });

      setCheckResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to check quality gates");
    } finally {
      setWorking(false);
    }
  }

  useEffect(() => {
    loadGates();
  }, []);

  return (
    <section id="quality-gates" className="quality-gate-manager">
      <div className="quality-gate-header">
        <div>
          <p className="eyebrow">Quality Gates</p>
          <h2>CI-style release checks for AI runs</h2>
          <p>
            Create and test metric rules that block unreliable RAG or agent outputs
            before they are treated as release-ready.
          </p>
        </div>

        <div className="quality-gate-actions">
          <button onClick={seedDefaults} disabled={working}>
            <ShieldCheck size={16} />
            Seed defaults
          </button>
          <button className="ghost-button" onClick={loadGates} disabled={working}>
            <RefreshCw size={16} />
            Refresh
          </button>
        </div>
      </div>

      {error ? (
        <div className="inline-error">
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      ) : null}

      <div className="quality-gate-layout">
        <div className="quality-gate-form-card">
          <h3>Create gate</h3>

          <label>
            Gate name
            <input
              value={form.name}
              onChange={(event) =>
                setForm((current) => ({
                  ...current,
                  name: event.target.value,
                }))
              }
            />
          </label>

          <label>
            Metric name
            <input
              value={form.metric_name}
              onChange={(event) =>
                setForm((current) => ({
                  ...current,
                  metric_name: event.target.value,
                }))
              }
            />
          </label>

          <div className="quality-gate-form-row">
            <label>
              Operator
              <select
                value={form.operator}
                onChange={(event) =>
                  setForm((current) => ({
                    ...current,
                    operator: event.target.value as GateOperator,
                  }))
                }
              >
                <option value=">=">&gt;=</option>
                <option value="<=">&lt;=</option>
                <option value=">">&gt;</option>
                <option value="<">&lt;</option>
                <option value="==">==</option>
              </select>
            </label>

            <label>
              Threshold
              <input
                type="number"
                step="0.01"
                value={form.threshold}
                onChange={(event) =>
                  setForm((current) => ({
                    ...current,
                    threshold: Number(event.target.value),
                  }))
                }
              />
            </label>
          </div>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={form.is_active}
              onChange={(event) =>
                setForm((current) => ({
                  ...current,
                  is_active: event.target.checked,
                }))
              }
            />
            Active gate
          </label>

          <button onClick={createGate} disabled={working}>
            <Plus size={16} />
            Create quality gate
          </button>
        </div>

        <div className="quality-gate-list-card">
          <div className="quality-gate-list-top">
            <h3>Configured gates</h3>
            <button className="ghost-button" onClick={runSampleCheck} disabled={working}>
              <CheckCircle2 size={16} />
              Test sample metrics
            </button>
          </div>

          {loading ? (
            <div className="inspection-state">
              <Loader2 className="spin" size={20} />
              <p>Loading quality gates...</p>
            </div>
          ) : gates.length === 0 ? (
            <p className="muted-text">
              No quality gates configured yet. Seed defaults to create the standard
              reliability rules.
            </p>
          ) : (
            <div className="quality-gate-list">
              {gates.map((gate) => (
                <GateCard
                  key={gate.id}
                  gate={gate}
                  onToggle={() => toggleGate(gate)}
                  onDelete={() => deleteGate(gate)}
                  disabled={working}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {checkResult ? (
        <div className="quality-gate-check-panel">
          <div className="quality-gate-check-top">
            <div>
              <h3>Sample gate check</h3>
              <p>
                Checked active gates against representative RAG reliability metrics.
              </p>
            </div>

            <span
              className={
                checkResult.gate_status === "passed"
                  ? "badge success"
                  : "badge danger"
              }
            >
              {checkResult.gate_status}
            </span>
          </div>

          <div className="quality-gate-check-summary">
            <span>{checkResult.passed_checks} passed</span>
            <span>{checkResult.failed_checks} failed</span>
            <span>{checkResult.total_checks} total</span>
          </div>

          <div className="check-list">
            {checkResult.checks.map((check) => (
              <div
                className={check.passed ? "check-card check-pass" : "check-card check-fail"}
                key={check.gate_id}
              >
                <div>
                  <strong>{check.name}</strong>
                  <p>
                    {check.metric_name} {check.operator} {check.threshold}
                  </p>
                </div>

                <div className="check-card-meta">
                  <span className={check.passed ? "badge success" : "badge danger"}>
                    {check.passed ? "passed" : "failed"}
                  </span>
                  <span>
                    actual {check.actual_value === null ? "—" : check.actual_value}
                  </span>
                </div>

                <p className="check-failure">{check.reason}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

function GateCard({
  gate,
  onToggle,
  onDelete,
  disabled,
}: {
  gate: QualityGate;
  onToggle: () => void;
  onDelete: () => void;
  disabled: boolean;
}) {
  return (
    <div className={gate.is_active ? "gate-card gate-active" : "gate-card gate-inactive"}>
      <div>
        <strong>{gate.name}</strong>
        <p>
          {gate.metric_name} {gate.operator} {gate.threshold}
        </p>
        <span>{gate.is_active ? "Active" : "Inactive"}</span>
      </div>

      <div className="gate-card-actions">
        <button className="ghost-button" onClick={onToggle} disabled={disabled}>
          {gate.is_active ? "Disable" : "Enable"}
        </button>
        <button className="danger-button" onClick={onDelete} disabled={disabled}>
          <Trash2 size={15} />
        </button>
      </div>
    </div>
  );
}