# AgentEvalOps Demo Script

## Demo Goal

This demo shows AgentEvalOps as an AI reliability and evaluation platform for RAG pipelines and tool-calling agents.
The goal is not to present it as a chatbot. The goal is to show that the system can:

1. ingest documents
2. retrieve evidence using different retrieval strategies
3. generate grounded RAG answers
4. inspect traces
5. evaluate answer quality
6. evaluate tool-calling agents
7. compare benchmark pipelines
8. apply quality gates
9. generate readiness reports

---

## 1. Pre-Demo Checklist

Before starting the demo, make sure these are running.

### Backend infrastructure

```bash
docker compose up -d
```

Expected services:

```text
PostgreSQL
Redis
Qdrant
```

### Backend API

```bash
uvicorn backend.app.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm run dev
```

Open:

```text
http://localhost:5173
```

### Validation checks

```bash
python -m pytest
```

Expected:

```text
109 passed
```

Frontend build check:

```bash
cd frontend
npm run build
```

---

## 2. Opening Pitch

Use this explanation:

```text
AgentEvalOps is an AI reliability platform for RAG pipelines and tool-calling agents.
Instead of only answering questions from documents, it evaluates whether the answer is grounded, cited, relevant, low-risk, and release-ready.
The system compares retrieval strategies, tracks traces, evaluates hallucination risk, measures latency and token usage, and applies CI-style quality gates before a pipeline is considered ready.
```

Avoid saying:

```text
This is a document chatbot.
```

Say:

```text
This is an AI reliability and evaluation platform for RAG and agent workflows.
```

---

## 3. Dashboard Walkthrough

Open:

```text
http://localhost:5173
```

Show the top dashboard.

Point out:

```text
Documents
Experiments
Runs
Benchmark runs
Run health
Latency and cost
Average quality
Faithfulness
Citation accuracy
Hallucination risk
```

Explain:

```text
The dashboard gives an aggregate reliability view across all local RAG, agent, benchmark, and experiment runs.
```

Current local snapshot:

```text
Documents: 8
Experiments: 1
Runs: 434
Benchmark runs: 50
Completed run rate: 99.77%
Average overall quality: 0.9344
Average faithfulness: 1.0000
Average hallucination risk: 0.0360
Average citation accuracy: 0.9824
```

---

## 4. Document Explorer Demo

Go to:

```text
Documents
```

Show:

```text
Uploaded/indexed documents
Document status
Chunk count
Chunk viewer
Delete action
```

Talking point:

```text
Documents are split into chunks, stored in PostgreSQL, embedded using Sentence Transformers, and indexed into Qdrant for dense retrieval.
```

If showing upload:

1. Upload a small `.md` or `.txt` policy file.
2. Wait for indexed status.
3. Open chunks.
4. Show chunk text and metadata.

Explain:

```text
This is the ingestion layer. It is separate from the RAG layer so that indexing, chunking, retrieval, and evaluation can be tested independently.
```

---

## 5. RAG Playground Demo

Go to:

```text
RAG Playground
```

Use this query:

```text
What must customers include when reporting a damaged product?
```

Recommended settings:

```text
Retrieval provider: hybrid
Top K: 3
Rerank: enabled
Candidate multiplier: 3
Quality gate profile: default-v1
```

Click run.

Show:

```text
Answer
Citations
Retrieved chunks
Latency
Token usage
Quality gate status
Failed gates, if any
Inspect run button
```

Talking point:

```text
This is not just returning an answer. The response is connected to retrieved source chunks, evaluated for reliability, and checked against quality gates.
```

Expected answer theme:

```text
Customers should include the order ID, photos of the damaged product, packaging photos, and a short issue description.
```

Explain:

```text
The answer is grounded in retrieved policy context and cites the source chunks used to generate the response.
```

---

## 6. Run Inspection Demo

After running a RAG query, click:

```text
Inspect Run
```

Show:

```text
Run metadata
Trace steps
Retrieved chunks
Evaluation results
Quality gate results
Latency
Tokens
Workflow status
```

Talking point:

```text
Every workflow creates a run record. The run inspection panel makes the RAG process debuggable by showing the internal steps instead of only showing the final answer.
```

Explain why this matters:

```text
If a RAG answer fails, we can inspect whether the issue came from retrieval, answer generation, citation validation, evaluation, or quality gates.
```

---

## 7. Agent Playground Demo

Go to:

```text
Agents
```

Use one of these queries.

### Calculator agent query

```text
Calculate (142080 - 120000) / 120000 * 100
```

Expected answer:

```text
18.4
```

### Mock API query

```text
Check order status for ORD-1001
```

Expected behavior:

```text
The agent should use the mock API tool and return the order status.
```

### Customer plan query

```text
What is the customer plan for CUST-1001?
```

Expected behavior:

```text
The agent should use the mock API tool and return the customer plan.
```

Show:

```text
Final answer
Tool calls
Tool outputs
Latency
Run ID
Inspect run option
```

Talking point:

```text
The agent layer tests tool-use reliability, not just document QA. The evaluator checks whether the correct tools were selected, whether tool calls succeeded, whether the order was correct, and whether the final answer was correct.
```

---

## 8. Experiment Comparison Demo

Go to:

```text
Compare
```

Show the benchmark comparison table.

Point out the compared pipelines:

```text
BM25 Baseline
Dense Retrieval
Hybrid Retrieval
Hybrid Retrieval + Rerank
```

Current latest benchmark comparison:

| Pipeline | Passed Cases | Failed Cases | Pass Rate | Overall Quality | Hallucination Risk | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|
| BM25 Baseline | 3/3 | 0 | 100.00% | 0.9993 | 0.0000 | 58.33 ms |
| Dense Retrieval | 2/3 | 1 | 66.67% | 0.9240 | 0.0000 | 3231.00 ms |
| Hybrid Retrieval | 3/3 | 0 | 100.00% | 1.0000 | 0.0000 | 76.67 ms |
| Hybrid Retrieval + Rerank | 3/3 | 0 | 100.00% | 1.0000 | 0.0000 | 918.67 ms |

Talking point:

```text
The point of this page is to compare pipeline behavior empirically. Dense retrieval is not assumed to be better. The benchmark shows how each retrieval strategy actually performs on the same dataset.
```

Explain the result:

```text
BM25, Hybrid, and Hybrid + Rerank passed all current benchmark cases. Dense Retrieval passed only 2 out of 3 cases. Hybrid + Rerank has the strongest release-readiness result because it passed all cases and all quality gates, though it adds reranking latency.
```

---

## 9. Benchmark Report Demo

From the comparison table or recent benchmark runs, open the report for:

```text
benchrun_d941b4e81fa6
```

Show:

```text
Readiness card
Aggregate gates
Case pass rate
Failed cases
Report context
Retrieval metrics
Answer quality
Latency, cost, and tokens
Quality gate checks
Failed items
Raw report details
```

Point out:

```text
Ready: true
Status: ready
Reasons: all_release_checks_passed
Recommendation: Ready to proceed to frontend/demo usage.
```

Best benchmark run:

| Field | Value |
|---|---|
| Pipeline | Hybrid Retrieval + Rerank |
| Passed cases | 3/3 |
| Failed cases | 0 |
| Pass rate | 100.00% |
| Overall quality | 1.0000 |
| Hallucination risk | 0.0000 |
| Average latency | 918.67 ms |
| Quality gates | 6/6 passed |
| Readiness status | ready |

Talking point:

```text
The report turns raw evaluation metrics into a release-readiness decision. This is the production reliability part of the project.
```

---

## 10. Retrieval Metrics Explanation

In the benchmark report, show:

```text
Recall@k
Precision@k
MRR
nDCG@k
```

Current latest result:

| Pipeline | Recall@k | Precision@k | MRR | nDCG@k |
|---|---:|---:|---:|---:|
| BM25 Baseline | 1.0000 | 0.3333 | 1.0000 | 1.0000 |
| Dense Retrieval | 1.0000 | 0.3333 | 1.0000 | 1.0000 |
| Hybrid Retrieval | 1.0000 | 0.3333 | 1.0000 | 1.0000 |
| Hybrid Retrieval + Rerank | 1.0000 | 0.3333 | 1.0000 | 1.0000 |

Explain:

```text
Recall@k tells us whether the expected supporting chunk was retrieved.
Precision@k tells us how much of the retrieved context was relevant.
MRR tells us how high the first relevant chunk appeared.
nDCG tells us ranking quality.
```

For this benchmark:

```text
All pipelines retrieved the expected relevant chunk.
Precision@k is 0.3333 because top_k is 3 and there is one expected relevant chunk per case.
```

---

## 11. Quality Gates Demo

Go to:

```text
Quality Gates
```

Show:

```text
Seed defaults
Create custom gate
Enable/disable gate
Delete gate
Test sample metrics
```

Click:

```text
Test sample metrics
```

Explain:

```text
Quality gates are CI-style rules for AI reliability. They check metrics like pass rate, failed case rate, hallucination risk, quality score, latency, and estimated cost.
```

Show the default gate logic:

| Gate | Rule |
|---|---|
| Minimum Benchmark Pass Rate | pass_rate >= 0.80 |
| Maximum Failed Case Rate | failed_case_rate <= 0.20 |
| Minimum Average Overall Quality | average_overall_quality_score >= 0.70 |
| Maximum Average Hallucination Risk | average_hallucination_risk <= 0.20 |
| Maximum Average Latency | average_latency_ms <= 15000 |
| Maximum Average Estimated Cost | average_estimated_cost <= 0.05 |

Talking point:

```text
This allows the system to block low-quality prompt, retrieval, or model changes before they are treated as ready.
```

---

## 12. Failed Run / Gate Failure Explanation

If showing an earlier failed benchmark run, use:

```text
benchrun_87210417ad13
```

Explain:

```text
This earlier benchmark run failed 1 out of 3 cases.
```

Failed gates:

| Failed Gate | Expected | Actual |
|---|---:|---:|
| Minimum Benchmark Pass Rate | >= 0.8000 | 0.6667 |
| Maximum Failed Case Rate | <= 0.2000 | 0.3333 |

Talking point:

```text
This is important because the system does not only show successful runs. It can reject weak runs and explain why they failed.
```

---

## 13. Swagger API Demo

Open:

```text
http://localhost:8000/docs
```

Show API groups:

```text
Documents
Retrieval
RAG
Agents
Evaluations
Experiments
Benchmarks
Pipeline Configs
Quality Gates
Runs
Reports
Dashboard
```

Talking point:

```text
The frontend is built on top of a proper FastAPI backend. The backend exposes separate API modules for ingestion, retrieval, RAG, agents, evaluation, quality gates, reports, and dashboard data.
```

---

## 14. Documentation Demo

Show these files:

```text
README.md
architecture.md
evaluation_methodology.md
benchmark_report.md
docs/demo_script.md
```

Explain:

```text
The documentation explains the project as an AI reliability platform, not just a coding demo. The benchmark report records actual evaluation results, and the methodology explains how retrieval, answer quality, hallucination, agent behavior, and quality gates are measured.
```

---

## 15. Best End-to-End Demo Flow

Use this order during a final walkthrough:

```text
1. Open dashboard
2. Show document explorer and indexed document chunks
3. Run RAG query in RAG Playground
4. Inspect the run trace
5. Run calculator or mock API agent query
6. Show agent tool calls
7. Open pipeline comparison
8. Open latest benchmark report
9. Show quality gates
10. Open benchmark_report.md
11. End with architecture and evaluation methodology
```

This order tells a complete story:

```text
Data enters the system.
The system retrieves evidence.
The system produces an answer.
The system traces the workflow.
The system evaluates the answer.
The system compares pipeline variants.
The system applies quality gates.
The system generates a readiness decision.
```

---

## 16. Short Demo Script

Use this spoken version for a 2-minute explanation:

```text
AgentEvalOps is an AI reliability platform for RAG pipelines and tool-calling agents.

The system ingests documents, chunks them, embeds them with Sentence Transformers, stores metadata in PostgreSQL, and stores vectors in Qdrant.

On top of that, it supports BM25 retrieval, dense vector retrieval, hybrid retrieval, and reranking. I can run a RAG query, inspect the retrieved chunks and citations, and then inspect the full trace of the workflow.

The key part is evaluation. The platform measures retrieval metrics like Recall@k, Precision@k, MRR, and nDCG, and answer metrics like support score, query relevance, faithfulness, hallucination risk, citation accuracy, latency, tokens, and cost.

It also supports tool-calling agents and evaluates whether the agent selected the right tool, used tools in the right order, avoided unnecessary tool calls, and produced the correct final answer.

For benchmark comparison, I compared BM25, Dense Retrieval, Hybrid Retrieval, and Hybrid Retrieval with reranking on the same benchmark dataset. The latest Hybrid + Rerank benchmark passed 3 out of 3 cases, achieved 1.0 overall quality, had 0.0 hallucination risk, and passed all 6 quality gates.

The quality gates are CI-style release checks. They block bad AI pipeline changes if pass rate, failed case rate, hallucination risk, quality, latency, or cost cross configured thresholds.

So this is not a chatbot. It is a reliability, evaluation, observability, and quality-gating platform for RAG and agent workflows.
```

---

## 17. Honest Limitations to Mention if Asked

Say this clearly if someone asks about scale:

```text
The current benchmark report uses a small validation dataset, so I am not claiming large-scale production performance yet. The focus of the project is the evaluation infrastructure: ingestion, retrieval comparison, metrics, traces, reports, and quality gates. The same system can be extended to larger datasets.
```

If asked about ML model training:

```text
The current version uses Sentence Transformers and reranking, but I am not claiming a fine-tuned custom ML model yet. That is listed as a future improvement.
```

If asked about hosted LLM comparison:

```text
The architecture supports provider abstraction, but the current benchmark report focuses on the reliability pipeline using the current configured answer generator. Hosted model comparison can be added as a future benchmark.
```

---

## 18. Common Questions and Answers

### Is this just a chatbot?

```text
No. A chatbot returns answers. AgentEvalOps evaluates whether those answers are reliable. It tracks retrieval quality, answer grounding, hallucination risk, citations, traces, benchmark results, and quality gates.
```

### Why do you compare BM25, dense, and hybrid retrieval?

```text
Because retrieval performance depends on the dataset. Dense retrieval is not always better. The system benchmarks retrieval strategies on the same test cases so the best configuration is chosen empirically.
```

### Why do quality gates matter?

```text
Quality gates turn metrics into release decisions. Instead of only saying the model scored 0.8, the system decides whether the pipeline should pass or fail based on configured thresholds.
```

### Why is reranking useful?

```text
Reranking can improve the order of retrieved chunks by scoring query-chunk relevance after the initial retrieval step. It can improve quality, but it also adds latency, so the benchmark helps decide whether the tradeoff is worth it.
```

### What happens when a benchmark fails?

```text
The system stores failed items, failure categories, metrics, and quality gate results. This makes it possible to debug whether the failure came from retrieval, answer generation, citation validation, or gate thresholds.
```

### What makes the project production-oriented?

```text
The production-oriented parts are modular backend APIs, persistent run and trace storage, benchmark datasets, retrieval comparison, evaluation metrics, quality gates, readiness reports, and dashboard visibility.
```

---

## 19. Demo Close

End with this:

```text
The main idea behind AgentEvalOps is that AI outputs should not be trusted just because they sound correct.
They should be traced, evaluated, compared, and gated.
That is what this platform does for RAG pipelines and tool-calling agents.
```