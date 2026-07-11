# AgentEvalOps Evaluation Methodology

## 1. Purpose

AgentEvalOps evaluates RAG pipelines and tool-calling agents using measurable reliability signals.
The goal is not only to generate an answer, but to answer these questions:

- Did the system retrieve the right evidence?
- Did the final answer use that evidence correctly?
- Did the answer avoid unsupported claims?
- Were citations valid?
- Did the agent choose and execute the correct tools?
- How much latency, token usage, and estimated cost did the workflow create?
- Should this run, experiment, or benchmark be marked ready?

This methodology explains how AgentEvalOps measures retrieval quality, answer quality, hallucination risk, agent behavior, operational performance, and quality gate readiness.

---

## 2. Evaluation Philosophy

AgentEvalOps follows four evaluation principles.

### 2.1 Evaluate intermediate steps, not only final answers

A final answer can be wrong for different reasons:

- retrieval missed the right chunk
- retrieval found the right chunk but ranked it poorly
- answer generation ignored the retrieved evidence
- citation validation failed
- a quality gate blocked the answer
- an agent chose the wrong tool
- a tool returned an incorrect or irrelevant result

Because of this, AgentEvalOps stores traces and evaluates each important stage of the workflow.

### 2.2 Compare pipeline configurations empirically

The system does not assume that BM25, dense retrieval, hybrid retrieval, or reranking is always best.
Instead, AgentEvalOps runs benchmark datasets across multiple pipeline configurations and compares:

- pass rate
- answerable accuracy
- retrieval quality
- answer quality
- hallucination risk
- latency
- token usage
- estimated cost
- quality gate result

### 2.3 Treat hallucination risk as a release blocker

A response that sounds good but is not supported by the retrieved evidence is unsafe.
AgentEvalOps evaluates whether the answer is grounded in the retrieved context and uses quality gates to block weak or risky outputs.

### 2.4 Convert metrics into decisions

Metrics alone are not enough. AgentEvalOps uses quality gates to convert evaluation results into pass/fail decisions.
A benchmark or experiment is marked ready only when required quality gates pass.

---

## 3. Evaluation Flow

The evaluation flow depends on whether the workflow is a RAG run, an agent run, or a benchmark run.

### 3.1 RAG Run Evaluation Flow

```text
User question
    ↓
Retrieve chunks
    ↓
Generate grounded answer
    ↓
Check citations
    ↓
Evaluate answer support
    ↓
Evaluate query-answer relevance
    ↓
Estimate hallucination risk
    ↓
Track latency and token usage
    ↓
Apply run-level quality gates
    ↓
Store evaluation results
```

### 3.2 Agent Run Evaluation Flow

```text
User task
    ↓
Agent selects tool
    ↓
Tool executes
    ↓
Agent observes output
    ↓
Agent produces final answer
    ↓
Evaluate tool selection
    ↓
Evaluate tool order
    ↓
Evaluate tool-call success
    ↓
Evaluate final answer correctness
    ↓
Track unnecessary tool calls
    ↓
Store evaluation results
```

### 3.3 Benchmark Evaluation Flow

```text
Benchmark dataset
    ↓
Test cases
    ↓
Pipeline configuration
    ↓
Run workflow for each case
    ↓
Evaluate every case
    ↓
Aggregate metrics
    ↓
Apply benchmark-level quality gates
    ↓
Generate benchmark report
    ↓
Generate readiness decision
```

---

## 4. Retrieval Evaluation

Retrieval evaluation checks whether the system found the correct supporting evidence.
This is important because a RAG system can only answer correctly if the correct source material is retrieved.

### 4.1 Inputs

Retrieval evaluation uses:

- user query
- retrieved chunks
- expected relevant chunk IDs
- top-k value

Example benchmark case:

```json
{
  "question": "What must customers include when reporting a damaged product?",
  "expected_relevant_chunk_ids": ["chunk_78f69d174504"]
}
```

### 4.2 Recall@k

Recall@k measures whether the expected relevant chunk appears anywhere in the top-k retrieved results.

```text
Recall@k = relevant retrieved chunks / total expected relevant chunks
```

Example:

```text
Expected relevant chunks: [chunk_12]
Retrieved top 3: [chunk_44, chunk_12, chunk_9]

Recall@3 = 1 / 1 = 1.0
```

A high Recall@k means the retriever successfully found the required evidence.

### 4.3 Precision@k

Precision@k measures how much of the retrieved context is relevant.

```text
Precision@k = relevant retrieved chunks / total retrieved chunks
```

Example:

```text
Expected relevant chunks: [chunk_12]
Retrieved top 3: [chunk_44, chunk_12, chunk_9]

Precision@3 = 1 / 3 = 0.3333
```

Precision@k is lower when the system retrieves extra chunks that are not expected relevant chunks.

### 4.4 MRR

MRR, or Mean Reciprocal Rank, measures how high the first relevant chunk appears in the retrieved result list.

```text
MRR = 1 / rank of first relevant chunk
```

Example:

```text
Expected relevant chunk appears at rank 2

MRR = 1 / 2 = 0.5
```

A higher MRR means the most useful evidence appears earlier.

### 4.5 nDCG@k

nDCG@k measures ranking quality by rewarding relevant chunks that appear higher in the result list.
A high nDCG@k means relevant evidence was not only retrieved, but ranked well.

### 4.6 Retrieval Metrics Used in Current Benchmark

The current benchmark computes retrieval metrics for cases with expected relevant chunk IDs.
Latest benchmark comparison:

| Pipeline | Recall@k | Precision@k | MRR | nDCG@k |
|---|---:|---:|---:|---:|
| BM25 Baseline | 1.0000 | 0.3333 | 1.0000 | 1.0000 |
| Dense Retrieval | 1.0000 | 0.3333 | 1.0000 | 1.0000 |
| Hybrid Retrieval | 1.0000 | 0.3333 | 1.0000 | 1.0000 |
| Hybrid Retrieval + Rerank | 1.0000 | 0.3333 | 1.0000 | 1.0000 |

In this benchmark, all retrieval pipelines found the expected relevant chunk. Precision@k is 0.3333 because the benchmark used top_k=3 and one expected relevant chunk per test case.

---

## 5. Answer Quality Evaluation

Answer quality evaluation checks whether the generated answer is useful, grounded, relevant, and safe.

### 5.1 Answer Support Score

Answer support score measures whether the final answer is supported by the retrieved chunks.
A high support score means the answer is grounded in retrieved context.

Signals used can include:

- overlap between answer and retrieved context
- source coverage
- cited chunk support
- retrieval confidence
- unsupported claim detection

### 5.2 Query-Answer Relevance Score

Query-answer relevance score measures whether the answer actually addresses the user’s question.
This prevents answers that are grounded but not directly useful.

Example:

```text
Question: How long do approved refunds usually take?

Good answer:
Approved refunds are completed within 5 to 7 business days.

Bad answer:
Customers must include order details when reporting damaged products.
```

The bad answer may be grounded in the policy document, but it does not answer the refund duration question.

### 5.3 Faithfulness Score

Faithfulness measures whether the answer stays consistent with the retrieved source context.
A faithful answer should not add facts that are not present in the retrieved chunks.

### 5.4 Hallucination Risk

Hallucination risk estimates the likelihood that the answer contains unsupported or fabricated information.
High hallucination risk can come from:

- weak retrieval scores
- unsupported claims
- missing citations
- answer content not found in retrieved context
- mismatch between answer and source chunks

### 5.5 Hallucination Rate

Hallucination rate tracks whether a run or benchmark produced hallucination-like failures.
It is useful as an aggregate metric across many runs or benchmark cases.

### 5.6 Citation Accuracy

Citation accuracy checks whether:

- the answer includes citations
- cited chunk IDs are valid
- cited chunks were actually retrieved
- citations refer to relevant supporting context

This prevents fake or irrelevant citations.

### 5.7 Overall Quality Score

Overall quality score is an aggregate answer quality signal.
It combines important quality dimensions such as:

- answer support
- query-answer relevance
- citation accuracy
- hallucination risk
- source coverage

The exact weighting can evolve, but the purpose is to give a single comparable score for pipeline ranking and gate checks.

---

## 6. Refusal and Weak-Evidence Handling

A reliable RAG system should not answer confidently when the retrieved evidence is weak.
AgentEvalOps supports safe fallback behavior when the answer does not satisfy reliability checks.

Example fallback:

```text
I could not find enough reliable evidence in the provided documents to answer this confidently.
```

This is important for unanswerable questions and low-confidence retrieval.

The system can treat blocked responses as successful when the correct behavior is refusal, depending on benchmark case type.

---

## 7. Agent Evaluation

Tool-calling agents introduce reliability risks that normal RAG evaluation does not cover.
An agent can fail even if individual tools work correctly.

Example failures:

- choosing the wrong tool
- calling tools in the wrong order
- skipping a required tool
- making unnecessary tool calls
- producing the wrong final answer after correct tool usage

AgentEvalOps evaluates these behaviors separately.

### 7.1 Tool Selection Accuracy

Tool selection accuracy checks whether the agent chose the correct tool for the task.

Example:

```text
Question: Calculate revenue growth from Q1 to Q2.
Expected tool: calculator
```

If the agent uses document search instead of calculator, tool selection should fail.

### 7.2 Tool Order Accuracy

Tool order accuracy checks whether tools were used in the expected sequence.

Example:

```text
Expected order:
1. SQL or mock API lookup for Q1 revenue
2. SQL or mock API lookup for Q2 revenue
3. Calculator
```

Incorrect ordering can indicate poor planning.

### 7.3 Tool-Call Success Rate

Tool-call success rate checks whether tool executions completed successfully.
Tool calls that error, timeout, or return unusable data reduce this score.

### 7.4 Tool Efficiency Score

Tool efficiency score rewards agents that solve the task without unnecessary steps.
An agent that calls five tools when one was enough should be penalized.

### 7.5 Final Answer Correctness

Final answer correctness checks whether the agent’s final answer matches the expected outcome.
For numeric answers, tolerance-based matching may be used.

### 7.6 Unnecessary Tool-Call Rate

Unnecessary tool-call rate measures how often the agent used tools that were not needed.
Lower is better.

### 7.7 Current Agent Evaluation Metrics

The current experiment report includes these agent evaluation results:

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

These metrics show that the current agent evaluation flow successfully evaluates tool use, answer correctness, and unnecessary tool calls.

---

## 8. Latency, Token, and Cost Evaluation

Reliability is not only about correctness. A pipeline also needs to be operationally practical.
AgentEvalOps tracks:

- total latency
- prompt tokens
- completion tokens
- total tokens
- estimated cost

### 8.1 Latency

Latency is tracked at the run level and aggregated across experiments or benchmark runs.
High latency may come from:

- reranking
- slow retrieval
- LLM calls
- tool calls
- large contexts
- inefficient pipeline settings

### 8.2 Token Usage

Token usage is tracked using:

```text
prompt_tokens
completion_tokens
total_tokens
```

This helps compare prompt and retrieval configurations.

A pipeline that retrieves too much context may produce unnecessary prompt tokens.

### 8.3 Estimated Cost

Estimated cost is tracked so that benchmark results can include both quality and cost.
In the current benchmark, cost is recorded as 0.0 because the evaluated answer generator is extractive/mock-style rather than a paid hosted LLM call.

---

## 9. Benchmark Case Evaluation

Each benchmark test case is evaluated individually.

A case can include:

- question
- expected answer or expected terms
- expected relevant chunk IDs
- expected citations
- answerability label
- metadata

### 9.1 Answerable Cases

For answerable cases, the expected behavior is to answer correctly using retrieved evidence.
An answerable case may fail if:

- answer is incorrect
- answer is unsupported
- retrieval misses required evidence
- quality gates block the response
- citation validation fails
- expected terms are missing

### 9.2 Unanswerable Cases

For unanswerable cases, the expected behavior is to refuse or avoid unsupported claims.
An unanswerable case may fail if:

- the system invents an answer
- the system cites irrelevant chunks
- the system does not trigger safe fallback behavior

### 9.3 Case-Level Stored Outputs

For each benchmark item, AgentEvalOps can store:

- actual answer
- pass/fail status
- failure reason
- retrieved chunks
- evaluation metrics
- latency
- quality gate result
- trace/run reference

This makes benchmark failures debuggable.

---

## 10. Benchmark Aggregation

After all benchmark cases run, AgentEvalOps aggregates metrics.

### 10.1 Aggregate Metrics

Benchmark-level metrics include:

- total cases
- passed cases
- failed cases
- pass rate
- failed case rate
- answerable accuracy
- unanswerable accuracy
- average answer support score
- average query-answer relevance score
- average hallucination risk
- average overall quality score
- average latency
- average prompt tokens
- average completion tokens
- average total tokens
- average estimated cost
- total estimated cost

### 10.2 Current Benchmark Aggregation

Latest Hybrid Retrieval + Rerank benchmark run:

| Metric | Value |
|---|---:|
| Total cases | 3 |
| Passed cases | 3 |
| Failed cases | 0 |
| Pass rate | 1.0000 |
| Failed case rate | 0.0000 |
| Answerable accuracy | 1.0000 |
| Average answer support score | 1.0000 |
| Average query-answer relevance score | 1.0000 |
| Average hallucination risk | 0.0000 |
| Average overall quality score | 1.0000 |
| Average latency | 918.6667 ms |
| Average prompt tokens | 192.0000 |
| Average completion tokens | 37.3333 |
| Average total tokens | 229.3333 |
| Average estimated cost | 0.0000 |
| Total estimated cost | 0.0000 |

---

## 11. Pipeline Comparison Methodology

Pipeline comparison evaluates multiple configurations on the same benchmark dataset.
Current compared pipelines:

| Pipeline | Retrieval Provider | Rerank |
|---|---|---|
| BM25 Baseline | bm25 | false |
| Dense Retrieval | dense | false |
| Hybrid Retrieval | hybrid | false |
| Hybrid Retrieval + Rerank | hybrid | true |

### 11.1 Ranking Criteria

Pipeline results can be compared using:

1. pass rate
2. failed case count
3. overall quality score
4. hallucination risk
5. retrieval metrics
6. latency
7. estimated cost

A good pipeline should not only pass cases. It should also remain grounded, cite evidence, avoid hallucination, and stay within latency/cost limits.

### 11.2 Current Pipeline Comparison

| Pipeline | Passed Cases | Failed Cases | Pass Rate | Overall Quality | Hallucination Risk | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|
| BM25 Baseline | 3/3 | 0 | 1.0000 | 0.9993 | 0.0000 | 58.3333 ms |
| Dense Retrieval | 2/3 | 1 | 0.6667 | 0.9240 | 0.0000 | 3231.0000 ms |
| Hybrid Retrieval | 3/3 | 0 | 1.0000 | 1.0000 | 0.0000 | 76.6667 ms |
| Hybrid Retrieval + Rerank | 3/3 | 0 | 1.0000 | 1.0000 | 0.0000 | 918.6667 ms |

### 11.3 Interpretation

BM25, Hybrid Retrieval, and Hybrid Retrieval + Rerank all passed the current benchmark.
Dense Retrieval passed only 2 out of 3 cases, showing why retrieval strategies should be benchmarked rather than assumed.
Hybrid Retrieval + Rerank produced the strongest release-readiness result because it passed all cases, achieved perfect answer quality on the benchmark, had zero hallucination risk, and passed all aggregate quality gates.

---

## 12. Quality Gate Methodology

Quality gates convert metrics into release decisions.

A quality gate defines:

- metric name
- operator
- threshold
- active/inactive status
- optional metadata

Example:

```json
{
  "name": "Maximum Average Hallucination Risk",
  "metric_name": "average_hallucination_risk",
  "operator": "<=",
  "threshold": 0.2,
  "is_active": true
}
```

### 12.1 Gate Evaluation

For each active gate:

1. read the target metric
2. compare actual value against threshold
3. mark the gate as passed or failed
4. record failure reason if needed

### 12.2 Aggregate Gate Result

The aggregate gate result includes:

- total gates
- passed gates
- failed gates
- pass rate
- overall passed boolean
- list of individual checks

### 12.3 Current Benchmark Quality Gates

Latest Hybrid Retrieval + Rerank benchmark run:

| Gate | Metric | Rule | Actual | Status |
|---|---|---:|---:|---|
| Minimum Benchmark Pass Rate | pass_rate | >= 0.8000 | 1.0000 | Passed |
| Maximum Failed Case Rate | failed_case_rate | <= 0.2000 | 0.0000 | Passed |
| Minimum Average Overall Quality | average_overall_quality_score | >= 0.7000 | 1.0000 | Passed |
| Maximum Average Hallucination Risk | average_hallucination_risk | <= 0.2000 | 0.0000 | Passed |
| Maximum Average Latency | average_latency_ms | <= 15000.0000 | 918.6667 | Passed |
| Maximum Average Estimated Cost | average_estimated_cost | <= 0.0500 | 0.0000 | Passed |

### 12.4 Readiness Decision

When quality gates pass, the report can produce a readiness decision.
Current best benchmark run readiness:

```json
{
  "ready": true,
  "status": "ready",
  "scope_type": "benchmark_run",
  "reasons": ["all_release_checks_passed"],
  "recommendation": "Ready to proceed to frontend/demo usage."
}
```

---

## 13. Failure Analysis Methodology

Failed benchmark items are analyzed so the system can explain what went wrong.
Potential failure categories include:

- retrieval missed expected evidence
- answer did not include expected information
- answer was blocked by quality gate
- answer was unsupported
- expected keyword mismatch
- citation failure
- unanswerable behavior failure
- latency or cost threshold failure

### 13.1 Failed Item Data

A failed item can include:

- question
- expected behavior
- actual answer
- failure reason
- metrics
- quality gate result
- response blocked status
- latency

### 13.2 Current Failure Analysis

The latest successful Hybrid Retrieval + Rerank benchmark run had:

```text
Failed items: 0
Failure categories: none
```

An earlier failed benchmark run had:

| Failed Gate | Expected | Actual |
|---|---:|---:|
| Minimum Benchmark Pass Rate | >= 0.8000 | 0.6667 |
| Maximum Failed Case Rate | <= 0.2000 | 0.3333 |

This shows that AgentEvalOps can reject weak benchmark runs even when some metrics, such as hallucination risk, look good.

---

## 14. Experiment-Level Evaluation

Experiments group multiple runs and aggregate their reliability metrics.
Experiment reports include:

- run health
- average latency
- token usage
- cost
- RAG evaluation metrics
- agent evaluation metrics
- quality gate result
- readiness decision

### 14.1 Current Experiment Summary

Experiment:

```text
Hybrid reranking experiment
```

Configuration:

| Field | Value |
|---|---|
| Experiment ID | exp_69df97a7a045 |
| Retriever | hybrid |
| LLM provider | extractive |
| LLM model | extractive-v1 |
| Prompt version | v1 |
| Chunking strategy | boundary-aware |
| Reranker enabled | true |

Run health:

| Metric | Value |
|---|---:|
| Total runs | 9 |
| Completed runs | 9 |
| Failed runs | 0 |
| Completed run rate | 1.0000 |
| Failed run rate | 0.0000 |
| Average latency | 5402.1111 ms |
| Average prompt tokens | 274.0000 |
| Average completion tokens | 56.0000 |
| Average total tokens | 330.0000 |
| Total estimated cost | 0.0000 |

Experiment-level gates passed 6 out of 6 checks.

---

## 15. Dashboard-Level Evaluation

The dashboard aggregates system-wide evaluation data across local runs.
Current dashboard snapshot:

| Metric | Value |
|---|---:|
| Documents | 8 |
| Experiments | 1 |
| Runs | 434 |
| Benchmark runs | 50 |
| Completed runs | 433 |
| Failed runs | 1 |
| Completed run rate | 0.9977 |
| Failed run rate | 0.0023 |
| Average latency | 817.4724 ms |
| Average overall quality score | 0.9344 |
| Average faithfulness score | 1.0000 |
| Average hallucination risk | 0.0360 |
| Average hallucination rate | 0.0750 |
| Average citation accuracy score | 0.9824 |
| Average quality gate pass rate | 0.9280 |

This provides a high-level view of system reliability across many local workflow executions.

---

## 16. What the Current Evaluation Proves

The current evaluation setup proves that AgentEvalOps can:

1. ingest and index documents
2. compare retrieval strategies
3. evaluate whether expected evidence was retrieved
4. generate grounded answers
5. measure answer support and query relevance
6. detect hallucination risk
7. validate citations
8. track latency and token usage
9. evaluate tool-calling agent behavior
10. aggregate benchmark and experiment metrics
11. apply quality gates
12. produce readiness decisions

This makes the project an AI reliability platform rather than a basic RAG chatbot.

---

## 17. Current Limitations

The current methodology has some limitations.

- The current benchmark report uses a small validation dataset.
- The benchmark focuses on one synthetic policy document.
- The answer generator in the current benchmark is extractive.
- Hosted LLM provider comparison is not the current benchmark focus.
- The project does not currently claim a fine-tuned ML model.
- More datasets are needed to make broad production-level claims.
- Human evaluation is not yet part of the workflow.
- Automated semantic correctness evaluation can be expanded further.

---

## 18. Future Evaluation Improvements

Possible improvements:

- add larger benchmark datasets
- add more unanswerable cases
- add multi-document benchmark cases
- add adversarial questions
- add hosted LLM judge evaluation
- add human review labels
- add prompt version comparison reports
- add model provider comparison reports
- add chunking strategy comparison reports
- add hallucination-risk classifier
- add drift monitoring for query and embedding distributions
- export benchmark reports as HTML or PDF
- integrate quality gates into CI pipelines

---

## 19. Summary

AgentEvalOps evaluates AI systems by tracing and measuring the full workflow, not just the final answer.
The methodology combines:

- retrieval metrics
- answer quality metrics
- hallucination checks
- citation checks
- agent tool-use metrics
- latency and cost tracking
- benchmark aggregation
- quality gates
- readiness decisions

The key idea is:

> A RAG or agent pipeline should not be trusted because it produced an answer. It should be trusted only after it retrieves the right evidence, produces a grounded response, passes evaluations, and satisfies release-quality gates.