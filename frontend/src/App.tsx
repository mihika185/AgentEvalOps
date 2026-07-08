import{ 
  useEffect, useState
} from "react";
import{
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Clock,
  Database,
  FileText,
  Gauge,
  GitBranch,
  Loader2,
  PlayCircle,
  ShieldCheck,
} from "lucide-react";

import{
  fetchJson
} from "./api";

import ReportPanel from "./ReportPanel";
import RunInspectionPanel from "./RunInspectionPanel";
import type{
  DashboardBenchmarkRun,
  DashboardExperiment,
  DashboardRun,
  DashboardSummary,
  ReportTarget,
} from "./types";
import RagPlayground from "./RagPlayground";

export default function App(){
  const [dashboard, setDashboard] = useState<DashboardSummary | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [selectedReport, setSelectedReport] = useState<ReportTarget | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function loadDashboard(){
    try {
      setLoading(true);
      setError(null);

      const data = await fetchJson<DashboardSummary>("/dashboard/summary");
      setDashboard(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dashboard");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadDashboard();
  }, []);

  if (loading) {
    return (
      <PageShell>
        <div className="center-card">
          <Loader2 className="spin" size={26} />
          <p>Loading AgentEvalOps dashboard...</p>
        </div>
      </PageShell>
    );
  }

  if (error || !dashboard) {
    return (
      <PageShell>
        <div className="center-card error-card">
          <AlertTriangle size={28} />
          <h2>Dashboard could not load</h2>
          <p>{error}</p>
          <button onClick={loadDashboard}>Retry</button>
        </div>
      </PageShell>
    );
  }

  return (
    <PageShell>
      <header className="hero">
        <div>
          <p className="eyebrow">AI Reliability Platform</p>
          <h1>AgentEvalOps Dashboard</h1>
          <p className="hero-subtitle">
            Monitor RAG quality, benchmark readiness, latency, cost, citations,
            hallucination risk, and experiment health from one place.
          </p>
        </div>

        <div className="hero-status">
          <ShieldCheck size={20} />
          <span>{dashboard.service_version}</span>
        </div>
      </header>
      <RagPlayground
        onInspectRun={(runId) => {
        setSelectedReport(null);
        setSelectedRunId(runId);
        }}
      />
      <section className="metric-grid">
        <MetricCard
          label="Documents"
          value={dashboard.counts.documents}
          icon={<FileText size={22} />}
        />
        <MetricCard
          label="Experiments"
          value={dashboard.counts.experiments}
          icon={<GitBranch size={22} />}
        />
        <MetricCard
          label="Runs"
          value={dashboard.counts.runs}
          icon={<PlayCircle size={22} />}
        />
        <MetricCard
          label="Benchmark Runs"
          value={dashboard.counts.benchmark_runs}
          icon={<Database size={22} />}
        />
      </section>

      <section className="panel-grid">
        <Panel title="Run Health" icon={<Activity size={20} />}>
          <ScoreRow
            label="Completed run rate"
            value={formatPercent(dashboard.run_health.completed_run_rate)}
          />
          <ScoreRow
            label="Failed run rate"
            value={formatPercent(dashboard.run_health.failed_run_rate)}
            danger={Number(dashboard.run_health.failed_run_rate || 0) > 0.05}
          />
          <ScoreRow
            label="Completed runs"
            value={formatNumber(dashboard.run_health.completed_runs)}
          />
          <ScoreRow
            label="Failed runs"
            value={formatNumber(dashboard.run_health.failed_runs)}
          />
        </Panel>

        <Panel title="Latency & Cost" icon={<Clock size={20} />}>
          <ScoreRow
            label="Average latency"
            value={formatMs(dashboard.latency_cost.average_latency_ms)}
          />
          <ScoreRow
            label="Max latency"
            value={formatMs(dashboard.latency_cost.max_latency_ms)}
          />
          <ScoreRow
            label="Average tokens"
            value={formatNumber(dashboard.latency_cost.average_total_tokens)}
          />
          <ScoreRow
            label="Total estimated cost"
            value={formatCurrency(dashboard.latency_cost.total_estimated_cost)}
          />
        </Panel>

        <Panel title="Quality" icon={<Gauge size={20} />}>
          <ScoreRow
            label="Overall quality"
            value={formatScore(dashboard.quality.average_overall_quality_score)}
          />
          <ScoreRow
            label="Faithfulness"
            value={formatScore(dashboard.quality.average_faithfulness_score)}
          />
          <ScoreRow
            label="Citation accuracy"
            value={formatScore(dashboard.quality.average_citation_accuracy_score)}
          />
          <ScoreRow
            label="Hallucination risk"
            value={formatScore(dashboard.quality.average_hallucination_risk)}
            danger={Number(dashboard.quality.average_hallucination_risk || 0) > 0.2}
          />
        </Panel>
      </section>

      <section className="content-grid">
        <Panel title="Recent Runs" icon={<BarChart3 size={20} />} wide>
          <p className="panel-hint">
            Click any run to inspect trace steps and evaluations.
          </p>
          <div className="table-list">
            {dashboard.recent_runs.map((run) => (
              <RunRow
                key={run.id}
                run={run}
                onSelect={() => {
                  setSelectedReport(null);
                  setSelectedRunId(run.id);
                }}
              />
            ))}
          </div>
        </Panel>

        <Panel title="Recent Experiments" icon={<GitBranch size={20} />}>
          <p className="panel-hint">
            Click an experiment to view aggregate readiness.
          </p>
          <div className="stack">
            {dashboard.recent_experiments.map((experiment) => (
              <ExperimentCard
                key={experiment.id}
                experiment={experiment}
                onSelect={() => {
                  setSelectedRunId(null);
                  setSelectedReport({
                    type: "experiment",
                    id: experiment.id,
                  });
                }}
              />
            ))}
          </div>
        </Panel>

        <Panel title="Recent Benchmark Runs" icon={<Database size={20} />} wide>
          <p className="panel-hint">
            Click any benchmark run to view aggregate quality gates.
          </p>
          <div className="table-list">
            {dashboard.recent_benchmark_runs.map((benchmarkRun) => (
              <BenchmarkRunRow
                key={benchmarkRun.id}
                benchmarkRun={benchmarkRun}
                onSelect={() => {
                  setSelectedRunId(null);
                  setSelectedReport({
                    type: "benchmark",
                    id: benchmarkRun.id,
                  });
                }}
              />
            ))}
          </div>
        </Panel>
      </section>

      {selectedRunId ? (
        <RunInspectionPanel
          runId={selectedRunId}
          onClose={() => setSelectedRunId(null)}
        />
      ) : null}

      {selectedReport ? (
        <ReportPanel
          target={selectedReport}
          onClose={() => setSelectedReport(null)}
        />
      ) : null}
    </PageShell>
  );
}

function PageShell({ children }: { children: React.ReactNode }) {
  return <main className="page">{children}</main>;
}

function MetricCard({
  label,
  value,
  icon,
}: {
  label: string;
  value: number;
  icon: React.ReactNode;
}) {
  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <div>
        <p>{label}</p>
        <strong>{formatNumber(value)}</strong>
      </div>
    </div>
  );
}

function Panel({
  title,
  icon,
  wide = false,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  wide?: boolean;
  children: React.ReactNode;
}) {
  return (
    <section className={wide ? "panel panel-wide" : "panel"}>
      <div className="panel-header">
        <div className="panel-title">
          {icon}
          <h2>{title}</h2>
        </div>
      </div>
      {children}
    </section>
  );
}

function ScoreRow({
  label,
  value,
  danger = false,
}: {
  label: string;
  value: string;
  danger?: boolean;
}) {
  return (
    <div className="score-row">
      <span>{label}</span>
      <strong className={danger ? "danger-text" : ""}>{value}</strong>
    </div>
  );
}

function RunRow({
  run,
  onSelect,
}: {
  run: DashboardRun;
  onSelect: () => void;
}) {
  return (
    <button className="row-card clickable-row" onClick={onSelect}>
      <div>
        <div className="row-title">{run.input_query}</div>
        <div className="row-subtitle">
          {run.workflow_type} · {run.retrieval_provider || "no retriever"} ·{" "}
          {run.answer_generator || "no generator"}
        </div>
      </div>

      <div className="row-meta">
        <StatusBadge status={run.status} passed={run.quality_gate_passed} />
        <span>{formatMs(run.latency_ms)}</span>
      </div>
    </button>
  );
}

function ExperimentCard({
  experiment,
  onSelect,
}: {
  experiment: DashboardExperiment;
  onSelect: () => void;
}) {
  return (
    <button className="small-card clickable-card" onClick={onSelect}>
      <div className="row-title">{experiment.name}</div>
      <p>{experiment.description || "No description provided."}</p>

      <div className="tag-row">
        <span>{experiment.retriever_type || "retriever unknown"}</span>
        <span>{experiment.llm_provider || "provider unknown"}</span>
        <span>{experiment.reranker_enabled ? "reranker on" : "reranker off"}</span>
      </div>

      <div className="gate-line">
        {experiment.latest_quality_gate_result?.overall_passed ? (
          <>
            <CheckCircle2 size={16} />
            Aggregate gates passed
          </>
        ) : (
          <>
            <AlertTriangle size={16} />
            No latest gate result
          </>
        )}
      </div>
    </button>
  );
}

function BenchmarkRunRow({
  benchmarkRun,
  onSelect,
}: {
  benchmarkRun: DashboardBenchmarkRun;
  onSelect: () => void;
}) {
  return (
    <button className="row-card clickable-row" onClick={onSelect}>
      <div>
        <div className="row-title">
          {benchmarkRun.pipeline_config_name || benchmarkRun.id}
        </div>
        <div className="row-subtitle">
          {benchmarkRun.passed_cases}/{benchmarkRun.total_cases} cases passed ·
          quality {formatScore(benchmarkRun.average_overall_quality_score)}
        </div>
      </div>

      <div className="row-meta">
        <StatusBadge
          status={benchmarkRun.status}
          passed={benchmarkRun.latest_quality_gate_result?.overall_passed}
        />
        <span>{formatPercent(benchmarkRun.pass_rate)}</span>
      </div>
    </button>
  );
}

function StatusBadge({
  status,
  passed,
}: {
  status: string;
  passed?: boolean | null;
}) {
  if (passed === true) {
    return <span className="badge success">passed</span>;
  }

  if (passed === false) {
    return <span className="badge danger">failed</span>;
  }

  return <span className="badge neutral">{status}</span>;
}

function formatNumber(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return new Intl.NumberFormat("en-US").format(Number(value));
}

function formatScore(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return Number(value).toFixed(3);
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

  return `${Math.round(Number(value))} ms`;
}

function formatCurrency(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `$${Number(value).toFixed(4)}`;
}