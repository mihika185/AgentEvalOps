# AgentEvalOps

AgentEvalOps is a backend system for evaluating, comparing, and debugging Retrieval-Augmented Generation pipelines.

It provides document ingestion, vector retrieval, grounded answer generation, trace logging, heuristic answer evaluation, quality gates, benchmark datasets, pipeline comparison, and failure analysis.

The goal of the project is not just to answer questions from documents, but to measure whether a RAG pipeline is reliable enough to trust.

## What this project does

AgentEvalOps supports:

* Uploading and indexing documents
* Splitting documents into retrievable chunks
* Embedding chunks with Sentence Transformers
* Storing vectors in Qdrant
* Running grounded RAG answer workflows
* Tracking workflow runs and trace steps
* Evaluating answer support, relevance, hallucination risk, retrieval confidence, and source coverage
* Applying configurable quality gates
* Creating benchmark datasets and test cases
* Comparing multiple pipeline configurations
* Analyzing benchmark failures
* Reproducing a complete benchmark with a seed script

## Tech stack

* Python
* FastAPI
* PostgreSQL
* SQLAlchemy
* Qdrant
* Redis
* Sentence Transformers
* Docker Compose
* Pytest

## Project structure

```text
backend/        FastAPI backend, database models, RAG workflow, evaluation logic
frontend/       Frontend assets
sample_data/    Sample benchmark documents
scripts/        Utility and seed scripts
tests/          Unit tests
docker-compose.yml
requirements.txt
pytest.ini
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start infrastructure services:

```bash
docker compose up -d
```

Start the backend:

```bash
uvicorn backend.app.main:app --reload
```

The API docs should be available at:

```text
http://localhost:8000/docs
```

## Run tests

```bash
pytest
```

Expected result:

```text
29 passed
```

## Reproduce the ApexCart benchmark

AgentEvalOps includes a reproducible synthetic enterprise-policy benchmark.

The seed script:

1. Uploads and indexes the ApexCart policy document if needed
2. Creates or reuses the benchmark dataset
3. Adds benchmark test cases only if missing
4. Ensures top-k-3 and top-k-5 pipeline configs exist
5. Runs a benchmark comparison
6. Prints the winning pipeline and score summary

Run:

```bash
python3 scripts/seed_apexcart_benchmark.py
```

Expected output:

```text
Reusing existing benchmark dataset: dataset_...
Reusing existing indexed document: doc_...
Benchmark test cases: 0 created, 12 skipped.

Comparison result:
Best pipeline: MiniLM Extractive top-k-5

Scores:
- MiniLM Extractive top-k-3: 11/12 passed, answerable=9/10, unanswerable=2/2, pass_rate=0.9167
- MiniLM Extractive top-k-5: 12/12 passed, answerable=10/10, unanswerable=2/2, pass_rate=1.0
```

## What the benchmark proves

The ApexCart benchmark contains 12 test cases:

* 10 answerable policy questions
* 2 intentionally unanswerable questions

The comparison shows that retrieval depth matters:

```text
Top-k-3: 11/12 passed
Top-k-5: 12/12 passed
```

The top-k-3 pipeline misses one relevant chunk for the renewal-payment retry question. The top-k-5 pipeline retrieves enough evidence and answers correctly.

This demonstrates that AgentEvalOps can compare RAG pipeline variants and identify which configuration performs better under benchmarked conditions.

## Quality gates

AgentEvalOps uses quality gates to block answers that are not sufficiently supported by retrieved evidence.

The system tracks metrics such as:

* answer support score
* query-answer relevance score
* hallucination risk
* top retrieval score
* source coverage score
* overall quality score
* source chunk count

Unsupported or weakly supported answers are replaced with a safe fallback response:

```text
I could not find enough reliable evidence in the provided documents to answer this confidently.
```

This is especially important for unanswerable questions, where the system should avoid inventing an answer.

## Pipeline comparison

Pipeline configs allow different RAG settings to be compared on the same benchmark.

Example configs:

```text
MiniLM Extractive top-k-3
MiniLM Extractive top-k-5
```

The benchmark comparison endpoint ranks pipelines by pass rate and then average overall quality.

## Failure analysis

AgentEvalOps can analyze failed benchmark runs to show whether a failure likely came from:

* retrieval missing the right chunk
* answer generation choosing the wrong sentence
* quality gates blocking the answer
* expected keyword mismatch
* unanswerable behavior failure

This makes the project useful for debugging RAG systems, not just running them.

## Why this project matters

Many RAG projects stop after building a chatbot. AgentEvalOps focuses on the engineering layer needed to evaluate and improve those systems.

It answers questions like:

* Did the system retrieve the right evidence?
* Was the answer grounded in the document?
* Did the system avoid answering unsupported questions?
* Which pipeline configuration performs better?
* Why did a benchmark case fail?

This makes the project closer to an AI evaluation and observability platform than a simple RAG demo.

## Current benchmark result

Current ApexCart benchmark result:

```text
Best pipeline: MiniLM Extractive top-k-5
Total cases: 12
Passed cases: 12
Answerable accuracy: 10/10
Unanswerable accuracy: 2/2
Pass rate: 1.0
```

## License

This project is licensed under the terms in the LICENSE file.