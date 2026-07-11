# AgentEvalOps

**AgentEvalOps** is an AI reliability and evaluation platform for RAG pipelines and tool-calling agents.

It is designed like an internal AI infrastructure tool, not a document chatbot. The system ingests documents, indexes knowledge bases, runs RAG and agent workflows, evaluates output quality, tracks traces, compares pipeline configurations, and applies CI-style quality gates before a pipeline is treated as release-ready.

---

## Why this project exists

Most RAG applications stop after answering questions from documents. That is not enough for production AI systems.

AI teams need to know:

- Did the system retrieve the right evidence?
- Was the final answer grounded in the retrieved context?
- Did the answer cite real source chunks?
- Did the system avoid unsupported or hallucinated answers?
- Which retrieval pipeline performs better?
- How much latency, token usage, and cost does each configuration add?
- Should this prompt/model/retrieval change be allowed to ship?

AgentEvalOps answers these questions by combining RAG, agent execution, evaluation metrics, observability, benchmark reporting, and quality gates.

---

## Core Features

### Document Ingestion

- Upload and index documents
- Extract and clean document text
- Split text into retrievable chunks
- Store document and chunk metadata in PostgreSQL
- Generate embeddings with Sentence Transformers
- Store vectors in Qdrant
- Delete documents with database and vector cleanup

### Retrieval

- BM25 keyword retrieval
- Dense vector retrieval with Qdrant
- Hybrid retrieval
- Cross-encoder reranking
- Document-level filtering
- Retrieval comparison across pipeline configurations

### RAG Answering

- Question answering over uploaded documents
- Context-grounded answer generation
- Source chunk citations
- Refusal behavior when evidence is weak
- Latency, token, and cost tracking
- Run-level quality gate evaluation

### Tool-Calling Agents

- Agent workflow execution
- Calculator tool
- Document search tool
- Mock API tool
- Tool-call traces
- Agent evaluation metrics such as tool selection accuracy, tool order accuracy, answer correctness, and unnecessary tool-call rate

### Evaluation Engine

AgentEvalOps evaluates both retrieval and answer quality.

Retrieval metrics:

- Recall@k
- Precision@k
- MRR
- nDCG@k

Answer quality metrics:

- Answer support score
- Query-answer relevance score
- Faithfulness score
- Hallucination risk
- Hallucination rate
- Citation accuracy
- Source coverage
- Overall quality score

Agent metrics:

- Tool selection accuracy
- Tool order accuracy
- Tool-call success rate
- Tool efficiency score
- Final answer correctness
- Unnecessary tool-call rate

System metrics:

- Latency
- Prompt tokens
- Completion tokens
- Total tokens
- Estimated cost
- Failure rate

### Experiment and Benchmark Tracking

- Create experiments
- Compare retrieval pipelines
- Compare benchmark runs
- Store benchmark datasets and test cases
- Analyze failed benchmark items
- Generate aggregate benchmark reports
- Produce readiness decisions

### Quality Gates

Quality gates convert metrics into pass/fail release checks.

Example rules:

- Benchmark pass rate must be at least 80%
- Failed case rate must be at most 20%
- Average overall quality must be at least 0.70
- Average hallucination risk must be at most 0.20
- Average latency must be below the configured threshold
- Average estimated cost must stay below the configured limit

A benchmark or experiment can be marked release-ready only when the configured gates pass.

---

## Architecture

```text
Documents / Benchmark Cases
        ↓
Document Ingestion
        ↓
Chunking + Embeddings + PostgreSQL + Qdrant
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

## Tech Stack

### Backend

- Python
- FastAPI
- SQLAlchemy
- PostgreSQL
- Redis
- Qdrant
- Sentence Transformers
- BM25 retrieval
- Cross-encoder reranking
- Pytest

### Frontend

- React
- TypeScript
- Vite
- CSS
- Lucide React icons

### Infrastructure

- Docker Compose
- PostgreSQL container
- Redis container
- Qdrant container

---

## Project Structure

```text
AgentEvalOps/
├── backend/
│   └── app/
│       ├── api/                  FastAPI route modules
│       ├── agents/               Tool-calling agent runtime and tools
│       ├── database/             SQLAlchemy models and DB connection
│       ├── evaluation/           RAG, retrieval, agent, and quality evaluators
│       ├── ingestion/            Document loading, chunking, embedding, indexing
│       ├── observability/        Run and trace logging
│       ├── rag/                  RAG pipeline and answer generation
│       ├── retrieval/            BM25, dense, hybrid, and reranking services
│       └── utils/                Shared helpers
│
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── RagPlayground.tsx
│       ├── AgentPlayground.tsx
│       ├── DocumentExplorer.tsx
│       ├── ExperimentComparison.tsx
│       ├── QualityGateManager.tsx
│       ├── ReportPanel.tsx
│       └── RunInspectionPanel.tsx
│
├── sample_data/
│   └── apexcart_customer_operations_policy.md
│
├── scripts/
│   ├── create_tables.py
│   ├── seed_apexcart_benchmark.py
│   └── generate_benchmark_report.py
│
├── tests/
├── benchmark_report.md
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Setup

### 1. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Infrastructure Services

```bash
docker compose up -d
```

This starts:

- PostgreSQL
- Redis
- Qdrant

### 4. Create Database Tables

```bash
python scripts/create_tables.py
```

### 5. Start the Backend

```bash
uvicorn backend.app.main:app --reload
```

Backend API:

```text
http://localhost:8000
```

Swagger docs:

```text
http://localhost:8000/docs
```

### 6. Start the Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend app:

```text
http://localhost:5173
```

---

## Environment Variables

The project uses `.env` for local configuration.

Example:

```env
ENVIRONMENT=development
DEBUG=true
API_PREFIX=/api/v1

DATABASE_URL=postgresql://agentevalops:agentevalops@localhost:5432/agentevalops
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333

DEFAULT_LLM_PROVIDER=mock
DEFAULT_LLM_MODEL=mock-llm

DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DEFAULT_EMBEDDING_PROVIDER=sentence-transformers

OPENAI_API_KEY=
GROQ_API_KEY=

MAX_UPLOAD_MB=25
DEFAULT_RETRIEVAL_TOP_K=5
MAX_RETRIEVAL_TOP_K=20

ENABLE_TRACE_LOGGING=true
```

---

## Running Tests

Backend tests:

```bash
python -m pytest
```

Current backend test result:

```text
109 passed
```

Frontend build:

```bash
cd frontend
npm run build
```

---

## Main API Areas

The backend exposes APIs for:

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

Important flows:

- Upload and index documents
- Retrieve chunks with BM25, dense, or hybrid retrieval
- Ask RAG questions with citations
- Run tool-calling agents
- Inspect run traces
- Create benchmark datasets and test cases
- Run benchmark comparisons
- Evaluate runs and benchmark results
- Apply quality gates
- Generate benchmark and experiment reports
- View dashboard summaries

---

## Frontend Pages and Panels

### Dashboard

Shows system-wide health:

- Documents
- Experiments
- Runs
- Benchmark runs
- Run health
- Latency
- Cost
- Quality
- Hallucination risk
- Citation accuracy

### RAG Playground

Runs document-grounded RAG queries and shows:

- Answer
- Citations
- Retrieved chunks
- Retrieval settings
- Latency
- Tokens
- Quality gate status
- Failed gates if any

### Agent Playground

Runs tool-calling agent workflows and shows:

- Final answer
- Tool calls
- Tool outputs
- Agent status
- Latency
- Run inspection link

### Document Explorer

Supports:

- Document upload
- Indexed document listing
- Chunk inspection
- Document deletion

### Experiment Comparison

Shows:

- Recent benchmark runs
- Pipeline comparison
- Pass rate
- Quality score
- Hallucination risk
- Latency
- Report drilldown

### Quality Gate Manager

Supports:

- Seeding default gates
- Creating custom gates
- Enabling/disabling gates
- Deleting gates
- Testing sample metrics against active gates

### Run Inspection Panel

Shows:

- Run metadata
- Trace steps
- Evaluation metrics
- Quality gate results
- Retrieved chunks and workflow details

### Report Panel

Shows:

- Benchmark readiness
- Aggregate quality gates
- Pass/fail summary
- Retrieval metrics
- Latency, token, and cost metrics
- Failed items
- Raw report details

---

## Current Benchmark Report

The current benchmark uses the **ApexCart Refund Policy Benchmark** dataset.

Latest comparison:

| Pipeline | Passed Cases | Failed Cases | Pass Rate | Overall Quality | Hallucination Risk | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|
| BM25 Baseline | 3/3 | 0 | 100.00% | 0.9993 | 0.0000 | 58.33 ms |
| Dense Retrieval | 2/3 | 1 | 66.67% | 0.9240 | 0.0000 | 3231.00 ms |
| Hybrid Retrieval | 3/3 | 0 | 100.00% | 1.0000 | 0.0000 | 76.67 ms |
| Hybrid Retrieval + Rerank | 3/3 | 0 | 100.00% | 1.0000 | 0.0000 | 918.67 ms |

Best benchmark run:

| Field | Value |
|---|---|
| Benchmark Run ID | benchrun_d941b4e81fa6 |
| Pipeline | Hybrid Retrieval + Rerank |
| Passed cases | 3/3 |
| Failed cases | 0 |
| Pass rate | 100.00% |
| Average overall quality | 1.0000 |
| Average hallucination risk | 0.0000 |
| Average latency | 918.67 ms |
| Quality gates | 6/6 passed |
| Readiness status | ready |

Full benchmark details are available in:

```text
benchmark_report.md
```

---

## Quality Gate Example

The best benchmark run passed all default quality gates:

| Gate | Rule | Actual | Status |
|---|---:|---:|---|
| Minimum Benchmark Pass Rate | >= 0.80 | 1.00 | Passed |
| Maximum Failed Case Rate | <= 0.20 | 0.00 | Passed |
| Minimum Average Overall Quality | >= 0.70 | 1.00 | Passed |
| Maximum Average Hallucination Risk | <= 0.20 | 0.00 | Passed |
| Maximum Average Latency | <= 15000 ms | 918.67 ms | Passed |
| Maximum Average Estimated Cost | <= 0.05 | 0.00 | Passed |

Readiness decision:

```json
{
  "ready": true,
  "status": "ready",
  "reasons": ["all_release_checks_passed"],
  "recommendation": "Ready to proceed to frontend/demo usage."
}
```

---

## Dashboard Snapshot

Current local dashboard state:

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

## Reproduce the ApexCart Benchmark

The sample data includes:

```text
sample_data/apexcart_customer_operations_policy.md
```

The benchmark can be seeded and rerun using:

```bash
python scripts/seed_apexcart_benchmark.py
```

This script creates or reuses:

- ApexCart policy document
- Benchmark dataset
- Benchmark test cases
- Pipeline configurations
- Benchmark comparison runs

---

## What This Project Demonstrates

AgentEvalOps demonstrates:

- Backend system design for AI reliability workflows
- RAG pipeline construction and evaluation
- Retrieval strategy comparison
- Vector search with Qdrant
- Structured benchmark execution
- LLM/agent observability through traces
- Hallucination and faithfulness evaluation
- CI-style quality gates for AI outputs
- Full-stack implementation with FastAPI and React
- Practical debugging workflows for failed AI responses

---

## Limitations

This project is still a local evaluation platform and should not be presented as a large-scale production deployment.

Current limitations:

- The current benchmark report uses a small validation dataset.
- The benchmark focuses on one synthetic policy document.
- The default answer generator used in the current benchmark is extractive.
- The project does not currently claim a fine-tuned ML model.
- Hosted LLM provider comparison is not the focus of the current benchmark report.
- More benchmark datasets are needed before making large-scale performance claims.

---

## Future Improvements

Potential extensions:

- Add larger benchmark datasets across multiple document types
- Add prompt-version registry and prompt comparison reports
- Add hosted LLM comparison across providers
- Add a trained hallucination-risk classifier
- Add a fine-tuned reranker training pipeline
- Add PDF-heavy benchmark datasets
- Add authentication and role-based access
- Add Prometheus/Grafana monitoring
- Add exportable HTML/PDF benchmark reports
- Add CI integration for automated quality gate checks

---

## License

This project is licensed under the terms in the `LICENSE` file.