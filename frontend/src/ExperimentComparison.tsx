import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  AlertTriangle,
  BarChart3,
  GitBranch,
  GitCompare,
  Loader2,
  RefreshCw,
  Trophy,
} from "lucide-react";

import { fetchJson } from "./api";
import type {
  BenchmarkRunComparisonRecord,
  ExperimentListResponse,
  ExperimentRecord,
  ReportTarget,
} from "./types";

type ExperimentComparisonProps = {
  onOpenReport: (target: ReportTarget) => void;
};

export default function ExperimentComparison({
  onOpenReport,
}: ExperimentComparisonProps) {
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);
  const [benchmarkRuns, setBenchmarkRuns] = useState<
    BenchmarkRunComparisonRecord[]
  >([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const rankedBenchmarkRuns = useMemo(
    () => rankBenchmarkRuns(benchmarkRuns),
    [benchmarkRuns]
  );

  const bestBenchmarkRun = rankedBenchmarkRuns[0] || null;

  async function loadComparisonData() {
    try {
      setLoading(true);
      setError(null);

      const [experimentData, benchmarkRunData] = await Promise.all([
        fetchJson<ExperimentListResponse>("/experiments?limit=50"),
        fetchJson<BenchmarkRunComparisonRecord[]>("/benchmarks/runs?limit=50"),
      ]);

      setExperiments(experimentData.experiments);
      setBenchmarkRuns(benchmarkRunData);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load comparison data"
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadComparisonData();
  }, []);

  return (
    <section id="experiment-comparison" className="experiment-comparison">
      <div className="comparison-header">
        <div>
          <p className="eyebrow">Experiment Comparison</p>
          <h2>Compare pipelines, experiments, and benchmark outcomes</h2>
          <p>
            Review how retrieval methods, reranking, answer generators, latency,
            cost, and quality metrics perform across recent benchmark runs.
          </p>
        </div>

        <button
          className="ghost-button"
          type="button"
          onClick={loadComparisonData}
        >
          <RefreshCw size={17} />
          Refresh
        </button>
      </div>

      {loading ? (
        <div className="comparison-state">
          <Loader2 className="spin" size={24} />
          <span>Loading experiment comparison...</span>
        </div>
      ) : error ? (
        <div className="rag-error">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      ) : (
        <>
          <div className="comparison-grid">
            <SummaryCard
              title="Experiments"
              value={formatNumber(experiments.length)}
              subtitle="Tracked configurations"
              icon={<GitBranch size={18} />}
            />
            <SummaryCard
              title="Benchmark Runs"
              value={formatNumber(benchmarkRuns.length)}
              subtitle="Recent comparison runs"
              icon={<BarChart3 size={18} />}
            />
            <SummaryCard
              title="Best Pipeline"
              value={
                bestBenchmarkRun
                  ? pipelineName(bestBenchmarkRun)
                  : "No benchmark runs"
              }
              subtitle={
                bestBenchmarkRun
                  ? `${formatPercent(bestBenchmarkRun.pass_rate)} pass rate`
                  : "Run a benchmark first"
              }
              icon={<Trophy size={18} />}
              wide
            />
          </div>

          <BenchmarkComparisonTable
            benchmarkRuns={rankedBenchmarkRuns}
            onOpenReport={onOpenReport}
          />

          <ExperimentTable
            experiments={experiments}
            onOpenReport={onOpenReport}
          />
        </>
      )}
    </section>
  );
}

function SummaryCard({
  title,
  value,
  subtitle,
  icon,
  wide = false,
}: {
  title: string;
  value: string;
  subtitle: string;
  icon: ReactNode;
  wide?: boolean;
}) {
  return (
    <div className={wide ? "comparison-card comparison-card-wide" : "comparison-card"}>
      <div className="comparison-card-icon">{icon}</div>
      <div>
        <span>{title}</span>
        <strong>{value}</strong>
        <p>{subtitle}</p>
      </div>
    </div>
  );
}

function BenchmarkComparisonTable({
  benchmarkRuns,
  onOpenReport,
}: {
  benchmarkRuns: BenchmarkRunComparisonRecord[];
  onOpenReport: (target: ReportTarget) => void;
}) {
  if (benchmarkRuns.length === 0) {
    return (
      <div className="comparison-panel">
        <PanelTitle icon={<GitCompare size={18} />} title="Pipeline benchmark comparison" />
        <p className="muted-text">
          No benchmark runs found yet. Run a benchmark comparison to populate this table.
        </p>
      </div>
    );
  }

  return (
    <div className="comparison-panel">
      <PanelTitle icon={<GitCompare size={18} />} title="Pipeline benchmark comparison" />

      <div className="comparison-table-wrapper">
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Pipeline</th>
              <th>Retriever</th>
              <th>Rerank</th>
              <th>Top K</th>
              <th>Pass Rate</th>
              <th>Quality</th>
              <th>Hallucination</th>
              <th>Latency</th>
              <th>Cost</th>
              <th>Report</th>
            </tr>
          </thead>
          <tbody>
            {benchmarkRuns.map((run, index) => (
              <tr key={run.id}>
                <td>
                  <div className="table-primary">
                    <strong>
                      {index === 0 ? "🏆 " : ""}
                      {pipelineName(run)}
                    </strong>
                    <span>{run.id}</span>
                  </div>
                </td>
                <td>{metadataString(run.metadata_json, "retrieval_provider", "—")}</td>
                <td>{metadataBoolean(run.metadata_json, "rerank")}</td>
                <td>{metadataNumber(run.metadata_json, "top_k") ?? "—"}</td>
                <td>{formatPercent(run.pass_rate)}</td>
                <td>{formatScore(run.average_overall_quality_score)}</td>
                <td>{formatScore(run.average_hallucination_risk)}</td>
                <td>{formatMs(run.average_latency_ms)}</td>
                <td>{formatCost(metadataNumber(run.metadata_json, "average_estimated_cost"))}</td>
                <td>
                  <button
                    className="table-action-button"
                    type="button"
                    onClick={() =>
                      onOpenReport({
                        type: "benchmark",
                        id: run.id,
                      })
                    }
                  >
                    Open
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ExperimentTable({
  experiments,
  onOpenReport,
}: {
  experiments: ExperimentRecord[];
  onOpenReport: (target: ReportTarget) => void;
}) {
  return (
    <div className="comparison-panel">
      <PanelTitle icon={<GitBranch size={18} />} title="Tracked experiments" />

      {experiments.length === 0 ? (
        <p className="muted-text">No experiments found yet.</p>
      ) : (
        <div className="comparison-table-wrapper">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Experiment</th>
                <th>Retriever</th>
                <th>Provider</th>
                <th>Model</th>
                <th>Prompt</th>
                <th>Chunking</th>
                <th>Reranker</th>
                <th>Runs</th>
                <th>Report</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((experiment) => (
                <tr key={experiment.id}>
                  <td>
                    <div className="table-primary">
                      <strong>{experiment.name}</strong>
                      <span>{experiment.id}</span>
                    </div>
                  </td>
                  <td>{experiment.retriever_type}</td>
                  <td>{experiment.llm_provider}</td>
                  <td>{experiment.llm_model}</td>
                  <td>{experiment.prompt_version}</td>
                  <td>{experiment.chunking_strategy}</td>
                  <td>{experiment.reranker_enabled ? "on" : "off"}</td>
                  <td>{formatNumber(experiment.run_count)}</td>
                  <td>
                    <button
                      className="table-action-button"
                      type="button"
                      onClick={() =>
                        onOpenReport({
                          type: "experiment",
                          id: experiment.id,
                        })
                      }
                    >
                      Open
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function PanelTitle({ icon, title }: { icon: ReactNode; title: string }) {
  return (
    <div className="mini-panel-title">
      {icon}
      <h3>{title}</h3>
    </div>
  );
}

function rankBenchmarkRuns(
  benchmarkRuns: BenchmarkRunComparisonRecord[]
): BenchmarkRunComparisonRecord[] {
  const bestRunByPipeline = new Map<string, BenchmarkRunComparisonRecord>();

  for (const run of benchmarkRuns) {
    const key = comparisonKey(run);
    const currentBest = bestRunByPipeline.get(key);

    if (!currentBest || isBetterBenchmarkRun(run, currentBest)) {
      bestRunByPipeline.set(key, run);
    }
  }

  return [...bestRunByPipeline.values()].sort((left, right) => {
    const leftPassRate = left.pass_rate ?? 0;
    const rightPassRate = right.pass_rate ?? 0;

    if (rightPassRate !== leftPassRate) {
      return rightPassRate - leftPassRate;
    }

    const leftQuality = left.average_overall_quality_score ?? 0;
    const rightQuality = right.average_overall_quality_score ?? 0;

    if (rightQuality !== leftQuality) {
      return rightQuality - leftQuality;
    }

    const leftHallucination = left.average_hallucination_risk ?? 1;
    const rightHallucination = right.average_hallucination_risk ?? 1;

    if (leftHallucination !== rightHallucination) {
      return leftHallucination - rightHallucination;
    }

    const leftLatency = left.average_latency_ms ?? Number.MAX_SAFE_INTEGER;
    const rightLatency = right.average_latency_ms ?? Number.MAX_SAFE_INTEGER;

    return leftLatency - rightLatency;
  });
}

function comparisonKey(run: BenchmarkRunComparisonRecord) {
  const pipelineConfigId = metadataString(
    run.metadata_json,
    "pipeline_config_id",
    ""
  );

  if (pipelineConfigId) {
    return pipelineConfigId;
  }

  return [
    pipelineName(run),
    metadataString(run.metadata_json, "retrieval_provider", "unknown"),
    metadataString(run.metadata_json, "answer_generator_provider", "unknown"),
    metadataNumber(run.metadata_json, "top_k") ?? "unknown",
    metadataBoolean(run.metadata_json, "rerank"),
  ].join("|");
}

function isBetterBenchmarkRun(
  candidate: BenchmarkRunComparisonRecord,
  currentBest: BenchmarkRunComparisonRecord
) {
  const candidatePassRate = candidate.pass_rate ?? 0;
  const currentPassRate = currentBest.pass_rate ?? 0;

  if (candidatePassRate !== currentPassRate) {
    return candidatePassRate > currentPassRate;
  }

  const candidateQuality = candidate.average_overall_quality_score ?? 0;
  const currentQuality = currentBest.average_overall_quality_score ?? 0;

  if (candidateQuality !== currentQuality) {
    return candidateQuality > currentQuality;
  }

  const candidateLatency = candidate.average_latency_ms ?? Number.MAX_SAFE_INTEGER;
  const currentLatency = currentBest.average_latency_ms ?? Number.MAX_SAFE_INTEGER;

  if (candidateLatency !== currentLatency) {
    return candidateLatency < currentLatency;
  }

  return new Date(candidate.started_at).getTime() >
    new Date(currentBest.started_at).getTime();
}

function pipelineName(run: BenchmarkRunComparisonRecord) {
  return metadataString(
    run.metadata_json,
    "pipeline_config_name",
    `Benchmark ${run.id}`
  );
}

function metadataString(
  metadata: Record<string, unknown>,
  key: string,
  fallback: string
): string {
  const value = metadata[key];

  if (typeof value === "string" && value.trim()) {
    return value;
  }

  return fallback;
}

function metadataNumber(
  metadata: Record<string, unknown>,
  key: string
): number | null {
  const value = metadata[key];

  if (typeof value === "number") {
    return value;
  }

  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);

    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }

  return null;
}

function metadataBoolean(metadata: Record<string, unknown>, key: string) {
  const value = metadata[key];

  if (typeof value === "boolean") {
    return value ? "on" : "off";
  }

  if (typeof value === "string") {
    return value;
  }

  return "—";
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatScore(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return Number(value).toFixed(4);
}

function formatMs(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `${Math.round(Number(value))} ms`;
}

function formatCost(value: number | null | undefined) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `$${Number(value).toFixed(4)}`;
}