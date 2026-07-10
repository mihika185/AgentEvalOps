# AgentEvalOps Benchmark Report

## Project

**AgentEvalOps** is an AI reliability platform for evaluating RAG pipelines and tool-calling agents. This report summarizes the current benchmark results for retrieval quality, answer quality, hallucination risk, latency, token usage, agent behavior, and CI-style quality gates.

The current benchmark is an initial validation benchmark using the **ApexCart Refund Policy Benchmark** dataset.

---

## 1. Benchmark Scope

### Dataset

| Field | Value |
|---|---|
| Dataset | ApexCart Refund Policy Benchmark |
| Dataset ID | dataset_33f2264a2f42 |
| Total test cases | 3 |
| Answerable cases | 3 |
| Unanswerable cases | 0 |
| Expected relevant chunks | Present for retrieval metric computation |
| Benchmark runner | benchmark-runner-v2 |

### Evaluated Pipelines

| Pipeline | Retrieval Provider | Rerank | Top K | Candidate Multiplier |
|---|---:|---:|---:|---:|
| BM25 Baseline | bm25 | false | 3 | 3 |
| Dense Retrieval | dense | false | 3 | 3 |
| Hybrid Retrieval | hybrid | false | 3 | 3 |
| Hybrid Retrieval + Rerank | hybrid | true | 3 | 3 |

### Runtime Configuration

| Component | Value |
|---|---|
| Answer generator provider | extractive |
| Answer generator model | simple-extractive-v1 |
| Embedding provider | sentence-transformers |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Quality gate profile | default-v1 |

---

## 2. Pipeline Comparison Results

Latest benchmark comparison:

| Pipeline | Passed Cases | Failed Cases | Pass Rate | Answerable Accuracy | Overall Quality | Hallucination Risk | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 Baseline | 3/3 | 0 | 100.00% | 100.00% | 0.9993 | 0.0000 | 58.33 ms |
| Dense Retrieval | 2/3 | 1 | 66.67% | 66.67% | 0.9240 | 0.0000 | 3231.00 ms |
| Hybrid Retrieval | 3/3 | 0 | 100.00% | 100.00% | 1.0000 | 0.0000 | 76.67 ms |
| Hybrid Retrieval + Rerank | 3/3 | 0 | 100.00% | 100.00% | 1.0000 | 0.0000 | 918.67 ms |

### Interpretation

BM25, Hybrid Retrieval, and Hybrid Retrieval + Rerank all passed the current benchmark with a 100% pass rate. Dense Retrieval underperformed on this dataset, passing only 2 out of 3 cases.

Hybrid Retrieval + Rerank achieved the strongest reliability profile because it passed all benchmark cases, achieved perfect answer quality on the current dataset, had zero hallucination risk, and passed all aggregate quality gates. It was slower than BM25 and plain Hybrid Retrieval because of reranking overhead.

BM25 performed very well on this small policy benchmark because the questions and source document contain strong lexical overlap. Dense Retrieval was weaker here, which shows why the platform needs empirical benchmark comparison instead of assuming semantic retrieval is always better.

---

## 3. Retrieval Quality Metrics

For cases with expected relevant chunk IDs, the benchmark computes Recall@k, Precision@k, MRR, and nDCG@k.

| Pipeline | Recall@k | Precision@k | MRR | nDCG@k |
|---|---:|---:|---:|---:|
| BM25 Baseline | 1.0000 | 0.3333 | 1.0000 | 1.0000 |
| Dense Retrieval | 1.0000 | 0.3333 | 1.0000 | 1.0000 |
| Hybrid Retrieval | 1.0000 | 0.3333 | 1.0000 | 1.0000 |
| Hybrid Retrieval + Rerank | 1.0000 | 0.3333 | 1.0000 | 1.0000 |

### Interpretation

All pipelines retrieved the expected relevant chunk within the top-k results for the latest benchmark run. The Precision@k value is 0.3333 because the benchmark uses top_k=3 and one expected relevant chunk per case. This means the relevant chunk was found, but the retrieved context still includes extra chunks.

The retrieval metrics validate that the benchmark runner is not only checking final answers. It also checks whether the retrieval layer found the expected supporting evidence.

---

## 4. Best Benchmark Run: Hybrid Retrieval + Rerank

| Field | Value |
|---|---|
| Benchmark Run ID | benchrun_d941b4e81fa6 |
| Pipeline | Hybrid Retrieval + Rerank |
| Status | completed |
| Passed cases | 3 |
| Failed cases | 0 |
| Pass rate | 100.00% |
| Failed case rate | 0.00% |
| Answerable accuracy | 100.00% |
| Average answer support score | 1.0000 |
| Average query-answer relevance score | 1.0000 |
| Average hallucination risk | 0.0000 |
| Average overall quality score | 1.0000 |
| Average latency | 918.67 ms |
| Average prompt tokens | 192.00 |
| Average completion tokens | 37.33 |
| Average total tokens | 229.33 |
| Average estimated cost | $0.0000 |
| Total estimated cost | $0.0000 |

### Readiness Decision

| Field | Value |
|---|---|
| Ready | true |
| Status | ready |
| Reason | all_release_checks_passed |
| Recommendation | Ready to proceed to frontend/demo usage. |

The best benchmark run had no failed benchmark items and was marked ready by the aggregate report generator.

---

## 5. Quality Gate Results

The best benchmark run was evaluated using the `default-v1` quality gate profile.

| Gate | Metric | Rule | Actual | Status |
|---|---|---:|---:|---|
| Minimum Benchmark Pass Rate | pass_rate | >= 0.8000 | 1.0000 | Passed |
| Maximum Failed Case Rate | failed_case_rate | <= 0.2000 | 0.0000 | Passed |
| Minimum Average Overall Quality | average_overall_quality_score | >= 0.7000 | 1.0000 | Passed |
| Maximum Average Hallucination Risk | average_hallucination_risk | <= 0.2000 | 0.0000 | Passed |
| Maximum Average Latency | average_latency_ms | <= 15000.0000 | 918.6667 | Passed |
| Maximum Average Estimated Cost | average_estimated_cost | <= 0.0500 | 0.0000 | Passed |

### Quality Gate Summary

| Field | Value |
|---|---:|
| Total gates | 6 |
| Passed gates | 6 |
| Failed gates | 0 |
| Gate pass rate | 100.00% |
| Overall passed | true |

### Interpretation

The benchmark run passed all release checks. This confirms that the platform can make CI-style pass/fail decisions for AI pipeline changes based on quality, hallucination risk, latency, cost, pass rate, and failed-case rate.

---

## 6. Failure Analysis

### Latest Successful Benchmark Run

| Field | Value |
|---|---|
| Failed items | 0 |
| Failure categories | None |

The latest Hybrid Retrieval + Rerank benchmark run had no failed items.

### Earlier Failed Benchmark Run

An earlier Hybrid Retrieval + Rerank run failed 1 out of 3 cases. Its aggregate quality gate result failed because:

| Failed Gate | Expected | Actual |
|---|---:|---:|
| Minimum Benchmark Pass Rate | >= 0.8000 | 0.6667 |
| Maximum Failed Case Rate | <= 0.2000 | 0.3333 |

This is useful because it demonstrates that the quality gate system can reject a benchmark run even when some individual metrics, such as hallucination risk and latency, look acceptable.

---

## 7. Experiment Report Summary

Experiment: **Hybrid reranking experiment**

| Field | Value |
|---|---|
| Experiment ID | exp_69df97a7a045 |
| Retriever | hybrid |
| LLM provider | extractive |
| LLM model | extractive-v1 |
| Prompt version | v1 |
| Chunking strategy | boundary-aware |
| Reranker enabled | true |

### Experiment Run Health

| Metric | Value |
|---|---:|
| Total runs | 9 |
| Completed runs | 9 |
| Failed runs | 0 |
| Completed run rate | 100.00% |
| Failed run rate | 0.00% |
| Average latency | 5402.11 ms |
| Average prompt tokens | 274.00 |
| Average completion tokens | 56.00 |
| Average total tokens | 330.00 |
| Total estimated cost | $0.0000 |

### Experiment Aggregate Quality Gates

| Gate | Metric | Rule | Actual | Status |
|---|---|---:|---:|---|
| Minimum Completed Run Rate | completed_run_rate | >= 0.9000 | 1.0000 | Passed |
| Maximum Failed Run Rate | failed_run_rate | <= 0.1000 | 0.0000 | Passed |
| Minimum Average Overall Quality | average_overall_quality_score | >= 0.7000 | 0.9500 | Passed |
| Maximum Average Hallucination Risk | average_hallucination_risk | <= 0.2000 | 0.0000 | Passed |
| Maximum Average Latency | average_latency_ms | <= 15000.0000 | 5402.1111 | Passed |
| Maximum Average Estimated Cost | average_estimated_cost | <= 0.0500 | 0.0000 | Passed |

### Experiment Readiness

| Field | Value |
|---|---|
| Ready | true |
| Status | ready |
| Reason | all_release_checks_passed |
| Recommendation | Ready to proceed to frontend/demo usage. |

---

## 8. Agent Evaluation Summary

The experiment report also includes tool-calling agent evaluation metrics.

| Agent Metric | Average |
|---|---:|
| Agent eval passed | 1.0000 |
| Answer correctness | 1.0000 |
| Max tool call score | 1.0000 |
| Overall agent score | 1.0000 |
| Tool call success rate | 1.0000 |
| Tool efficiency score | 1.0000 |
| Tool order accuracy | 1.0000 |
| Tool selection accuracy | 1.0000 |
| Unnecessary tool call rate | 0.0000 |

### Interpretation

The agent evaluation validates that the platform is not limited to RAG. It also evaluates tool-calling behavior, including tool selection, tool call success, tool ordering, final answer correctness, and unnecessary tool usage.

---

## 9. Dashboard-Level System Snapshot

The dashboard currently tracks documents, experiments, RAG runs, benchmark runs, latency, quality, hallucination risk, citation accuracy, and quality gate outcomes.

| Metric | Value |
|---|---:|
| Documents | 8 |
| Experiments | 1 |
| Runs | 434 |
| Benchmark runs | 50 |
| Completed runs | 433 |
| Failed runs | 1 |
| Completed run rate | 99.77% |
| Failed run rate | 0.23% |
| Average latency | 817.47 ms |
| Average overall quality score | 0.9344 |
| Average faithfulness score | 1.0000 |
| Average hallucination risk | 0.0360 |
| Average hallucination rate | 0.0750 |
| Average citation accuracy score | 0.9824 |
| Average quality gate pass rate | 0.9280 |

---

## 10. Limitations

This benchmark is intentionally small and should be treated as an initial validation benchmark, not a large-scale production evaluation.

Current limitations:

1. The benchmark dataset has only 3 cases.
2. The current benchmark focuses on one refund policy dataset.
3. The answer generator used in this benchmark is extractive, not a full hosted LLM.
4. The current report does not claim fine-tuned model training.
5. More diverse documents and larger evaluation sets are needed before making broad production claims.

---

## 11. Conclusion

The current benchmark validates the core AgentEvalOps reliability workflow:

1. Documents are indexed into searchable chunks.
2. Multiple retrieval strategies are compared.
3. RAG answers are evaluated with support, relevance, hallucination, citation, latency, token, and cost metrics.
4. Benchmark runs produce aggregate reports.
5. Quality gates convert metrics into release-readiness decisions.
6. Tool-calling agents are evaluated using agent-specific metrics.

The strongest current configuration is **Hybrid Retrieval + Rerank**, which achieved a 100% pass rate, 1.0000 overall quality score, 0.0000 hallucination risk, and passed all 6 aggregate quality gates on the current benchmark.

Future work should expand the number and diversity of benchmark cases before claiming large-scale production evaluation.