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

export type RunInspection = {
  run: RunDetail;
  trace_steps: TraceStep[];
  evaluation_results: EvaluationResult[];
  quality_gate_results: EvaluationResult[];
  summary: Record<string, unknown>;
};

export type RunDetail = {
  id: string;
  experiment_id?: string | null;
  workflow_type: string;
  status: string;
  input_query: string;
  output_answer?: string | null;
  latency_ms: number | null;
  error_message?: string | null;
  created_at: string;
  completed_at: string | null;
  metadata_json?: Record<string, unknown> | null;
};

export type TraceStep = {
  id?: string;
  step_index: number;
  step_name?: string | null;
  step_type?: string | null;
  status?: string | null;
  latency_ms?: number | null;
  input_data?: Record<string, unknown> | null;
  output_data?: Record<string, unknown> | null;
  error_message?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  metadata_json?: Record<string, unknown> | null;
};

export type EvaluationResult = {
  id?: string;
  evaluator_type: string;
  metric_name: string;
  metric_value: number;
  details?: Record<string, unknown> | null;
  created_at?: string | null;
};

export type ReportTarget = {
  type: "experiment" | "benchmark";
  id: string;
};

export type AggregateReport = {
  report_type?: string;
  service_version?: string;
  generated_at?: string;
  summary: Record<string, unknown>;
  quality_gate_result: AggregateQualityGateResult | null;
  readiness_decision: Record<string, unknown>;
  details: Record<string, unknown>;
};

export type AggregateQualityGateResult = {
  scope_type: string;
  scope_id: string;
  profile_name: string;
  evaluator_type: string;
  overall_passed: boolean;
  passed_count: number;
  failed_count: number;
  total_gates: number;
  pass_rate: number;
  aggregate_metrics: Record<string, number>;
  checks: AggregateQualityGateCheck[];
};

export type AggregateQualityGateCheck = {
  gate_id: string;
  gate_name: string;
  metric_name: string;
  metric_value: number;
  operator: string;
  threshold: number;
  passed: boolean;
  failure_reason?: string | null;
};