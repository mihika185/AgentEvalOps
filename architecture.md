# AgentEvalOps Architecture

## 1. Overview

AgentEvalOps is an AI reliability and evaluation platform for RAG pipelines and tool-calling agents.
The system is designed to answer a question that normal document chatbots do not answer:

> Is this RAG or agent pipeline reliable enough to trust?
To do that, AgentEvalOps combines document ingestion, retrieval, grounded answer generation, tool-calling agents, trace logging, evaluation metrics, benchmark comparisons, report generation, and CI-style quality gates.

The project is intentionally structured like an internal AI infrastructure platform rather than a chatbot UI.

---

## 2. High-Level System Flow

```text
Documents / Benchmark Cases
        ↓
Document Ingestion
        ↓
Text Cleaning + Chunking
        ↓
Embeddings + PostgreSQL + Qdrant
        ↓
BM25 / Dense / Hybrid Retrieval
        ↓
RAG Pipeline or Tool-Calling Agent
        ↓
Answer + Citations + Tool Calls
        ↓
Trace Logging
        ↓
Evaluation Engine
        ↓
Metrics + Failure Analysis
        ↓
Quality Gates
        ↓
Benchmark Report + Readiness Decision
        ↓
Frontend Dashboard
```

---

## 3. Main System Components

AgentEvalOps is organized into these major backend systems:

1. Document ingestion system
2. Retrieval system
3. RAG answering system
4. Tool-calling agent system
5. Evaluation engine
6. Experiment and benchmark tracking system
7. Quality gate system
8. Observability and trace logging system
9. Frontend dashboard

Each system is built as a separate module so the project stays extensible and does not collapse into one large chatbot-style script.

---

## 4. Backend Architecture

The backend is built with FastAPI and exposes versioned API routes under `/api/v1`.

```text
backend/app/
├── api/                  FastAPI route handlers
├── agents/               Tool-calling agent runtime and tools
├── database/             SQLAlchemy models and database connection
├── evaluation/           RAG, retrieval, agent, benchmark, and gate evaluators
├── ingestion/            Document loading, cleaning, chunking, embedding, indexing
├── observability/        Run logging and trace logging
├── rag/                  RAG workflow and answer generation
├── retrieval/            BM25, dense, hybrid retrieval, and reranking
└── utils/                Shared helpers
```

### Backend Responsibilities

The backend is responsible for:

- accepting document uploads
- extracting and chunking document text
- generating embeddings
- indexing chunks into Qdrant
- storing metadata and traces in PostgreSQL
- running RAG workflows
- running tool-calling agent workflows
- evaluating runs
- comparing benchmark pipelines
- enforcing quality gates
- generating dashboard and report data

---

## 5. Infrastructure Architecture

AgentEvalOps uses Docker Compose for local infrastructure.

```text
Docker Compose
├── PostgreSQL
├── Redis
└── Qdrant
```

### PostgreSQL

PostgreSQL stores structured application data:

- documents
- document chunks
- experiments
- runs
- trace steps
- evaluation results
- quality gates
- benchmark datasets
- benchmark test cases
- benchmark runs
- pipeline configurations

### Qdrant

Qdrant stores dense vector embeddings for document chunks.
It supports semantic retrieval by allowing query embeddings to be compared against indexed chunk embeddings.

### Redis

Redis is included as infrastructure for caching, future queueing, and runtime coordination. The current system is primarily database-driven, but Redis keeps the architecture ready for heavier async workflows.

---

## 6. Database Design

The database is the system of record for runs, traces, metrics, benchmark results, and quality decisions.

### Core Entities

```text
Document
    ↓
DocumentChunk
```

Documents represent uploaded source files. Chunks represent smaller searchable pieces of document text.

```text
Experiment
    ↓
Run
    ↓
TraceStep
    ↓
EvaluationResult
```

Experiments group related RAG or agent runs. Runs represent individual workflow executions. Trace steps record what happened inside each run. Evaluation results store metrics generated for that run.

```text
BenchmarkDataset
    ↓
BenchmarkTestCase
    ↓
BenchmarkRun
    ↓
BenchmarkRunItem
```

Benchmark datasets contain test cases. Benchmark runs execute one pipeline configuration against a dataset. Benchmark run items store case-level results.

```text
PipelineConfig
QualityGate
```

Pipeline configs describe retrieval and reranking settings. Quality gates define pass/fail rules over metrics.

---

## 7. Document Ingestion System

The ingestion system turns uploaded documents into searchable knowledge bases.

### Flow

```text
Upload file
    ↓
Load document text
    ↓
Clean text
    ↓
Split into chunks
    ↓
Generate embeddings
    ↓
Store chunks in PostgreSQL
    ↓
Store vectors in Qdrant
    ↓
Mark document as indexed
```

### Key Responsibilities

- read supported document files
- normalize extracted text
- split text into chunks
- preserve chunk metadata
- generate embeddings with Sentence Transformers
- persist chunks in PostgreSQL
- index vectors in Qdrant
- expose document indexing status

### Why this matters

RAG answer quality depends heavily on the quality of ingestion. Bad chunking or missing metadata can cause retrieval failures even when the source document contains the correct answer.

AgentEvalOps treats ingestion as a first-class pipeline step rather than a hidden preprocessing script.

---

## 8. Retrieval Architecture

The retrieval layer supports multiple retrieval strategies so pipelines can be compared empirically.

### Retrieval Methods

```text
BM25 Retrieval
Dense Vector Retrieval
Hybrid Retrieval
Hybrid Retrieval + Reranking
```

### BM25 Retrieval

BM25 is the lexical baseline. It works well when query words overlap with document text.

Strengths:

- fast
- explainable
- strong for policy-style documents with exact terminology

Weaknesses:

- weaker for paraphrased queries
- does not capture semantic similarity as well as embeddings

### Dense Retrieval

Dense retrieval uses embeddings and Qdrant vector search.

Flow:

```text
Query
    ↓
Query embedding
    ↓
Qdrant vector search
    ↓
Top-k semantic chunks
```

Strengths:

- captures semantic similarity
- useful when wording differs between query and document

Weaknesses:

- can retrieve semantically related but less exact chunks
- depends heavily on embedding model quality

### Hybrid Retrieval

Hybrid retrieval combines lexical and semantic retrieval.

```text
BM25 score + Dense score
        ↓
Combined ranking
```

This reduces dependence on only one retrieval strategy.

### Reranking

Reranking reorders candidate chunks after initial retrieval.

Flow:

```text
Initial candidate chunks
        ↓
Cross-encoder relevance scoring
        ↓
Final ranked chunks
```

Reranking adds latency, but can improve ranking quality when the candidate set contains multiple plausible chunks.

---

## 9. RAG Answering Architecture

The RAG system generates answers using retrieved document context.

### RAG Flow

```text
User question
    ↓
Retrieve relevant chunks
    ↓
Optionally rerank chunks
    ↓
Build grounded answer context
    ↓
Generate answer
    ↓
Attach citations
    ↓
Evaluate answer
    ↓
Apply quality gates
    ↓
Store run and traces
    ↓
Return answer
```

### Output

A RAG response includes:

- final answer
- source chunks
- citations
- retrieval scores
- latency
- token usage
- estimated cost
- quality gate result
- failed quality gates if any
- run ID for inspection

### Refusal and Blocking Behavior

If the retrieved context is weak or the answer fails quality checks, the system can avoid returning an unsafe answer.
This is important because production RAG systems should not confidently answer when they do not have enough evidence.

---

## 10. Tool-Calling Agent Architecture

AgentEvalOps includes a tool-calling agent layer so the platform is not limited to document QA.

### Agent Flow

```text
User task
    ↓
Agent planner
    ↓
Tool selection
    ↓
Tool execution
    ↓
Observation
    ↓
Repeat until final answer or max steps
    ↓
Final answer
    ↓
Trace + evaluation
```

### Supported Tool Types

The current agent system supports practical tools such as:

- calculator tool
- document search tool
- mock API tool

These tools allow the platform to test whether an agent chooses the correct tool, calls tools in the correct order, avoids unnecessary tool calls, and produces the correct final answer.

### Agent Evaluation

Agent runs are evaluated using metrics such as:

- tool selection accuracy
- tool order accuracy
- tool call success rate
- tool efficiency score
- final answer correctness
- unnecessary tool call rate
- overall agent score

---

## 11. Evaluation Engine

The evaluation engine is the core of AgentEvalOps.
It converts raw RAG and agent outputs into measurable reliability signals.

### Retrieval Metrics

For benchmark cases with expected relevant chunks, the system computes:

| Metric | Purpose |
|---|---|
| Recall@k | Did retrieval include the expected relevant chunk? |
| Precision@k | How much of the retrieved context was relevant? |
| MRR | How high was the first relevant chunk ranked? |
| nDCG@k | How good was the ranking quality? |

### Answer Quality Metrics

The system evaluates whether the generated answer is supported and relevant.

| Metric | Purpose |
|---|---|
| Answer support score | Measures whether the answer is supported by retrieved context |
| Query-answer relevance score | Measures whether the answer directly addresses the query |
| Faithfulness score | Measures whether the answer stays grounded |
| Hallucination risk | Estimates unsupported answer risk |
| Hallucination rate | Tracks hallucination-like failures |
| Citation accuracy | Checks whether citations refer to valid source chunks |
| Source coverage | Measures use of retrieved sources |
| Overall quality score | Aggregates answer quality signals |

### System Metrics

The platform also tracks operational metrics:

- latency
- prompt tokens
- completion tokens
- total tokens
- estimated cost
- run status
- failure rate

### Agent Metrics

Agent-specific metrics evaluate tool-use quality:

- correct tool selection
- correct tool ordering
- tool-call success
- answer correctness
- unnecessary tool-call rate
- overall agent score

---

## 12. Benchmark Architecture

Benchmarks are used to compare different pipeline configurations on the same dataset.

### Benchmark Flow

```text
Benchmark dataset
    ↓
Test cases
    ↓
Pipeline configuration
    ↓
Run RAG or agent workflow per case
    ↓
Evaluate each case
    ↓
Aggregate metrics
    ↓
Apply quality gates
    ↓
Generate report
```

### Pipeline Configurations

Pipeline configs can vary settings such as:

- retrieval provider
- top-k
- reranking enabled or disabled
- candidate multiplier
- quality gate profile

This allows direct comparison of BM25, dense retrieval, hybrid retrieval, and hybrid retrieval with reranking.

---

## 13. Quality Gate Architecture

Quality gates make the system behave like CI/CD for AI pipelines.
Instead of only showing metrics, the platform decides whether a run, experiment, or benchmark should pass.

### Example Gate Logic

```text
pass_rate >= 0.80
failed_case_rate <= 0.20
average_overall_quality_score >= 0.70
average_hallucination_risk <= 0.20
average_latency_ms <= 15000
average_estimated_cost <= 0.05
```

### Gate Flow

```text
Metrics
    ↓
Active quality gates
    ↓
Each gate checks one metric
    ↓
Passed and failed checks are collected
    ↓
Overall gate result is computed
    ↓
Readiness decision is generated
```

### Why Quality Gates Matter

Quality gates make the project stronger because they turn evaluation into a deployment decision.
The system is not only saying:

> The model got this score.

It is saying:

> This pipeline should or should not be treated as ready.

That is the difference between a chatbot demo and an AI reliability platform.

---

## 14. Observability and Trace Logging

Every important workflow produces trace data.

### Traceable Steps

A run can include trace steps such as:

- retrieval
- reranking
- prompt/context construction
- answer generation
- tool call
- citation check
- evaluation
- quality gate check

### Run Inspection

The frontend can inspect individual runs and show:

- input query
- workflow type
- status
- latency
- retrieved chunks
- generated answer
- trace steps
- evaluation metrics
- quality gate results

This makes failures debuggable instead of opaque.

---

## 15. Report Generation

AgentEvalOps generates aggregate reports for benchmark runs and experiments.

### Report Contents

A benchmark report includes:

- summary metrics
- case pass rate
- failed case rate
- answerable accuracy
- retrieval metrics
- answer quality metrics
- latency and token metrics
- quality gate checks
- failed items
- readiness decision

### Readiness Decision

A report produces a structured readiness result:

```json
{
  "ready": true,
  "status": "ready",
  "reasons": ["all_release_checks_passed"],
  "recommendation": "Ready to proceed to frontend/demo usage."
}
```

This makes the report useful for engineering decision-making, not just metric display.

---

## 16. Frontend Architecture

The frontend is built with React, TypeScript, and Vite.
It acts as an internal dashboard for AI reliability workflows.

### Main Frontend Components

```text
App.tsx
RagPlayground.tsx
AgentPlayground.tsx
DocumentExplorer.tsx
ExperimentComparison.tsx
QualityGateManager.tsx
ReportPanel.tsx
RunInspectionPanel.tsx
```

### Frontend Responsibilities

The frontend allows users to:

- upload and inspect documents
- run RAG queries
- run tool-calling agents
- inspect runs
- compare benchmark pipelines
- view benchmark and experiment reports
- manage quality gates
- test sample metrics against gates
- view system-wide dashboard metrics

---

## 17. API Architecture

The backend exposes grouped API modules.

```text
/health
/documents
/retrieval
/rag
/agents
/evaluations
/experiments
/benchmarks
/pipeline-configs
/quality-gates
/runs
/reports
/dashboard
```

### Important API Workflows

#### Document Upload and Indexing

```text
POST /documents/upload-and-index
GET  /documents
GET  /documents/{document_id}
GET  /documents/{document_id}/chunks
DELETE /documents/{document_id}
```

#### Retrieval

```text
POST /retrieval/search
POST /retrieval/compare
```

#### RAG

```text
POST /rag/answer
```

#### Agents

```text
POST /agents/run
```

#### Benchmarks

```text
POST /benchmarks/datasets
POST /benchmarks/datasets/{dataset_id}/test-cases
POST /benchmarks/datasets/{dataset_id}/compare
GET  /benchmarks/runs
GET  /benchmarks/runs/{benchmark_run_id}
```

#### Quality Gates

```text
POST   /quality-gates/defaults
POST   /quality-gates
GET    /quality-gates
GET    /quality-gates/{gate_id}
PATCH  /quality-gates/{gate_id}
DELETE /quality-gates/{gate_id}
POST   /quality-gates/check
```

#### Reports

```text
GET /reports/benchmark-runs/{benchmark_run_id}
GET /reports/experiments/{experiment_id}
```

#### Dashboard

```text
GET /dashboard/summary
```

---

## 18. Current Benchmark Architecture Validation

The current benchmark validates the core architecture using the ApexCart Refund Policy Benchmark.
Latest pipeline comparison:

| Pipeline | Passed Cases | Pass Rate | Overall Quality | Hallucination Risk | Avg Latency |
|---|---:|---:|---:|---:|---:|
| BM25 Baseline | 3/3 | 100.00% | 0.9993 | 0.0000 | 58.33 ms |
| Dense Retrieval | 2/3 | 66.67% | 0.9240 | 0.0000 | 3231.00 ms |
| Hybrid Retrieval | 3/3 | 100.00% | 1.0000 | 0.0000 | 76.67 ms |
| Hybrid Retrieval + Rerank | 3/3 | 100.00% | 1.0000 | 0.0000 | 918.67 ms |

The best benchmark run passed all quality gates:

| Gate Area | Result |
|---|---|
| Benchmark pass rate | Passed |
| Failed case rate | Passed |
| Average quality | Passed |
| Hallucination risk | Passed |
| Latency | Passed |
| Estimated cost | Passed |

This confirms that the system can evaluate, compare, and approve RAG pipeline configurations using measurable criteria.

---

## 19. Design Decisions

### Why separate retrieval strategies?

Different datasets behave differently. A semantic retriever is not always better than BM25. AgentEvalOps compares retrieval strategies directly instead of assuming one method is best.

### Why store traces?

RAG and agent failures are difficult to debug without intermediate steps. Trace logging makes it possible to inspect what the system retrieved, what tools were called, what answer was produced, and why a quality gate passed or failed.

### Why quality gates?

Metrics alone are passive. Quality gates turn metrics into decisions. This makes the system closer to production AI evaluation infrastructure.

### Why benchmark reports?

Benchmark reports create a reproducible record of pipeline performance. They make the project useful for comparing prompt, retrieval, model, and reranking changes.

### Why include agents?

Tool-calling agents introduce new reliability risks beyond RAG, such as wrong tool selection, bad tool order, and unnecessary tool calls. AgentEvalOps evaluates those behaviors separately.

---

## 20. Current Limitations

AgentEvalOps is currently a local evaluation platform.

Known limitations:

- The current benchmark report uses a small validation dataset.
- The benchmark focuses on one synthetic policy document.
- The current benchmark uses an extractive answer generator.
- The project does not currently claim a fine-tuned ML model.
- Hosted LLM provider comparison is not the focus of the current benchmark.
- Authentication and user management are not implemented.
- Large-scale async job orchestration is not yet implemented.
- More datasets are needed for broader evaluation claims.

---

## 21. Future Architecture Improvements

Possible future extensions:

- larger benchmark datasets
- prompt version registry
- hosted LLM provider comparison
- fine-tuned reranker training pipeline
- hallucination-risk classifier
- PDF-heavy benchmark suites
- async benchmark workers
- Prometheus and Grafana monitoring
- authentication and role-based access
- exportable HTML/PDF benchmark reports
- CI integration to automatically fail bad prompt or retrieval changes

---

## 22. Summary

AgentEvalOps is structured as an AI reliability platform with separate systems for ingestion, retrieval, RAG, agents, evaluation, quality gates, reports, and dashboarding.
The key architectural idea is simple:

> Do not just generate AI answers. Measure, trace, compare, and gate them.

This makes the system closer to production AI infrastructure than a traditional RAG chatbot.