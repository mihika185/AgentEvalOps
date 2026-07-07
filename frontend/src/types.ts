export type DashboardSummary = {
  service_version: string;
  generated_at: string;
  counts: {
    documents: number;
    experiments: number;
    runs: number;
    benchmark_runs: number;
  };
  run_health: Record<string, number>;
  latency_cost: Record<string, number | null>;
  quality: Record<string, number | null>;
  recent_runs: DashboardRun[];
  recent_experiments: DashboardExperiment[];
  recent_benchmark_runs: DashboardBenchmarkRun[];
};

export type DashboardRun = {
  id: string;
  experiment_id: string | null;
  workflow_type: string;
  status: string;
  input_query: string;
  latency_ms: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  estimated_cost: number | null;
  quality_gate_passed: boolean | null;
  response_blocked_by_quality_gate: boolean | null;
  retrieval_provider: string | null;
  answer_generator: string | null;
  created_at: string;
  completed_at: string | null;
};

export type DashboardExperiment = {
  id: string;
  name: string;
  description: string | null;
  retriever_type: string | null;
  llm_provider: string | null;
  llm_model: string | null;
  prompt_version: string | null;
  chunking_strategy: string | null;
  reranker_enabled: boolean;
  latest_quality_gate_result: {
    overall_passed: boolean;
    pass_rate: number;
  } | null;
  created_at: string;
};

export type DashboardBenchmarkRun = {
  id: string;
  dataset_id: string;
  status: string;
  total_cases: number;
  passed_cases: number;
  failed_cases: number;
  pass_rate: number;
  average_overall_quality_score: number | null;
  average_hallucination_risk: number | null;
  average_latency_ms: number | null;
  pipeline_config_id: string | null;
  pipeline_config_name: string | null;
  latest_quality_gate_result: {
    overall_passed: boolean;
    pass_rate: number;
  } | null;
  started_at: string;
  completed_at: string | null;
};