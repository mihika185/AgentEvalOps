# AgentEvalOps Benchmark Report

Generated at: `2026-07-05T19:28:18.889447+00:00`

## Scope

- Dataset filter: `dataset_c0b9622b9072`
- Experiment filter: `exp_69df97a7a045`
- Benchmark runs included: `10`

## Experiment Summary

- Experiment ID: `exp_69df97a7a045`
- Name: Hybrid reranking experiment
- Total runs: `7`
- Completed runs: `7`
- Failed runs: `0`
- Average latency: `3113.86 ms`

### Runs by Workflow

| Workflow Type | Count |
|---|---:|
| `rag_answer` | 1 |
| `tool_calling_agent` | 6 |

### Runs by Status

| Status | Count |
|---|---:|
| `completed` | 7 |

### Experiment Metric Highlights

| Evaluator | Metric | Count | Average | Min | Max |
|---|---|---:|---:|---:|---:|
| `heuristic-agent-evaluator-v1` | `answer_correctness` | 5 | 1 | 1 | 1 |
| `heuristic-agent-evaluator-v1` | `overall_agent_score` | 5 | 1 | 1 | 1 |
| `heuristic-agent-evaluator-v1` | `tool_selection_accuracy` | 5 | 1 | 1 | 1 |
| `heuristic-rag-evaluator-v1` | `answer_support_score` | 1 | 1 | 1 | 1 |
| `heuristic-rag-evaluator-v1` | `hallucination_risk` | 1 | 0 | 0 | 0 |
| `heuristic-rag-evaluator-v1` | `overall_quality_score` | 1 | 0.95 | 0.95 | 0.95 |
| `heuristic-rag-evaluator-v1` | `query_answer_relevance_score` | 1 | 0.8 | 0.8 | 0.8 |
| `quality-gate-evaluator-v1` | `quality_gate_overall_pass` | 1 | 1 | 1 | 1 |
| `quality-gate-evaluator-v1` | `quality_gate_pass_rate` | 1 | 1 | 1 | 1 |

## Benchmark Run Summary

| Benchmark Run | Pipeline | Status | Cases | Pass Rate | Avg Quality | Avg Latency | Recall@k | Precision@k | MRR | nDCG@k |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `benchrun_33f747cf7a44` | Hybrid Cross-Encoder Rerank top-k-3 | `completed` | 12/12 | 1 | 0.9423 | 370 | 1 | 0.3333 | 1 | 1 |
| `benchrun_6e7c7f8e1431` | Hybrid Extractive top-k-3 | `completed` | 12/12 | 1 | 0.9333 | 851.9167 | 1 | 0.3333 | 1 | 1 |
| `benchrun_ca7c8bc180fc` | Hybrid Cross-Encoder Rerank top-k-3 | `completed` | 12/12 | 1 | 0.9423 | 443.6667 | - | - | - | - |
| `benchrun_9f970f92ef61` | Hybrid Extractive top-k-3 | `completed` | 12/12 | 1 | 0.9333 | 888.9167 | - | - | - | - |
| `benchrun_9a54fb856029` | Hybrid Extractive top-k-5 | `completed` | 12/12 | 1 | 0.9305 | 106.25 | - | - | - | - |
| `benchrun_29d3bf1d181d` | BM25 Extractive top-k-5 | `completed` | 12/12 | 1 | 0.9363 | 49.1667 | - | - | - | - |
| `benchrun_2e74b4c3f74b` | MiniLM Extractive top-k-5 | `completed` | 12/12 | 1 | 0.8592 | 836.5 | - | - | - | - |
| `benchrun_2cbff58687c3` | Hybrid Extractive top-k-5 | `completed` | 12/12 | 1 | 0.9305 | 91.8333 | - | - | - | - |
| `benchrun_95ba9caecb95` | BM25 Extractive top-k-5 | `completed` | 12/12 | 1 | 0.9363 | 45.8333 | - | - | - | - |
| `benchrun_5320b0c1387d` | MiniLM Extractive top-k-5 | `completed` | 12/12 | 1 | 0.8592 | 857.9167 | - | - | - | - |

## Best Benchmark Run

- Benchmark run: `benchrun_33f747cf7a44`
- Dataset: `ApexCart Policy Reliability Benchmark`
- Pipeline: Hybrid Cross-Encoder Rerank top-k-3
- Pass rate: `1`
- Average quality: `0.9423`
- Average latency: `370 ms`
- Average recall@k: `1`
- Average MRR: `1`
- Average nDCG@k: `1`

Selection rule: highest pass rate, then highest average overall quality.

## Benchmark Run Details

### `benchrun_33f747cf7a44` — Hybrid Cross-Encoder Rerank top-k-3

- Dataset: `ApexCart Policy Reliability Benchmark`
- Status: `completed`
- Cases: `12/12`
- Pass rate: `1`
- Average overall quality: `0.9423`

| Result | Quality | Recall | MRR | Latency | Question |
|---|---:|---:|---:|---:|---|
| `PASS` | 0.9063 | - | - | 2605 | How long does a customer have to request a refund for a physical product? |
| `PASS` | 0.9583 | - | - | 140 | When does a monthly subscription cancellation take effect? |
| `PASS` | 0.9688 | - | - | 149 | How many times will ApexCart retry a failed renewal payment? |
| `PASS` | 0.9 | - | - | 136 | What is the usual delivery time for express shipping? |
| `PASS` | 0.95 | 1 | 1 | 138 | What must customers provide when reporting a damaged product? |
| `PASS` | 0.9 | - | - | 127 | What does the hardware accessory warranty exclude? |
| `PASS` | 0.9063 | - | - | 158 | How long does account deletion take after identity verification? |
| `PASS` | 0.9643 | - | - | 156 | For how long may ApexCart retain tax and fraud review records? |
| `PASS` | 0.9688 | - | - | 199 | What happens if a customer files a chargeback before contacting support? |
| `PASS` | 1 | - | - | 219 | When can a customer request escalation for an unresolved support issue? |
| `PASS` | 0.875 | - | - | 212 | Does ApexCart accept cryptocurrency payments for subscriptions? |
| `PASS` | 0.43 | - | - | 201 | Does ApexCart offer a student discount for annual plans? |

### `benchrun_6e7c7f8e1431` — Hybrid Extractive top-k-3

- Dataset: `ApexCart Policy Reliability Benchmark`
- Status: `completed`
- Cases: `12/12`
- Pass rate: `1`
- Average overall quality: `0.9333`

| Result | Quality | Recall | MRR | Latency | Question |
|---|---:|---:|---:|---:|---|
| `PASS` | 0.9063 | - | - | 9252 | How long does a customer have to request a refund for a physical product? |
| `PASS` | 0.9083 | - | - | 109 | When does a monthly subscription cancellation take effect? |
| `PASS` | 0.9472 | - | - | 107 | How many times will ApexCart retry a failed renewal payment? |
| `PASS` | 0.8817 | - | - | 108 | What is the usual delivery time for express shipping? |
| `PASS` | 0.95 | 1 | 1 | 78 | What must customers provide when reporting a damaged product? |
| `PASS` | 0.9 | - | - | 78 | What does the hardware accessory warranty exclude? |
| `PASS` | 0.9063 | - | - | 102 | How long does account deletion take after identity verification? |
| `PASS` | 0.9643 | - | - | 65 | For how long may ApexCart retain tax and fraud review records? |
| `PASS` | 0.9688 | - | - | 65 | What happens if a customer files a chargeback before contacting support? |
| `PASS` | 1 | - | - | 70 | When can a customer request escalation for an unresolved support issue? |
| `PASS` | 0.5045 | - | - | 103 | Does ApexCart accept cryptocurrency payments for subscriptions? |
| `PASS` | 0.4106 | - | - | 86 | Does ApexCart offer a student discount for annual plans? |

### `benchrun_ca7c8bc180fc` — Hybrid Cross-Encoder Rerank top-k-3

- Dataset: `ApexCart Policy Reliability Benchmark`
- Status: `completed`
- Cases: `12/12`
- Pass rate: `1`
- Average overall quality: `0.9423`

| Result | Quality | Recall | MRR | Latency | Question |
|---|---:|---:|---:|---:|---|
| `PASS` | 0.9063 | - | - | 3057 | How long does a customer have to request a refund for a physical product? |
| `PASS` | 0.9583 | - | - | 335 | When does a monthly subscription cancellation take effect? |
| `PASS` | 0.9688 | - | - | 179 | How many times will ApexCart retry a failed renewal payment? |
| `PASS` | 0.9 | - | - | 176 | What is the usual delivery time for express shipping? |
| `PASS` | 0.95 | - | - | 162 | What must customers provide when reporting a damaged product? |
| `PASS` | 0.9 | - | - | 163 | What does the hardware accessory warranty exclude? |
| `PASS` | 0.9063 | - | - | 226 | How long does account deletion take after identity verification? |
| `PASS` | 0.9643 | - | - | 172 | For how long may ApexCart retain tax and fraud review records? |
| `PASS` | 0.9688 | - | - | 205 | What happens if a customer files a chargeback before contacting support? |
| `PASS` | 1 | - | - | 165 | When can a customer request escalation for an unresolved support issue? |
| `PASS` | 0.875 | - | - | 245 | Does ApexCart accept cryptocurrency payments for subscriptions? |
| `PASS` | 0.43 | - | - | 239 | Does ApexCart offer a student discount for annual plans? |

### `benchrun_9f970f92ef61` — Hybrid Extractive top-k-3

- Dataset: `ApexCart Policy Reliability Benchmark`
- Status: `completed`
- Cases: `12/12`
- Pass rate: `1`
- Average overall quality: `0.9333`

| Result | Quality | Recall | MRR | Latency | Question |
|---|---:|---:|---:|---:|---|
| `PASS` | 0.9063 | - | - | 9630 | How long does a customer have to request a refund for a physical product? |
| `PASS` | 0.9083 | - | - | 167 | When does a monthly subscription cancellation take effect? |
| `PASS` | 0.9472 | - | - | 108 | How many times will ApexCart retry a failed renewal payment? |
| `PASS` | 0.8817 | - | - | 109 | What is the usual delivery time for express shipping? |
| `PASS` | 0.95 | - | - | 77 | What must customers provide when reporting a damaged product? |
| `PASS` | 0.9 | - | - | 91 | What does the hardware accessory warranty exclude? |
| `PASS` | 0.9063 | - | - | 126 | How long does account deletion take after identity verification? |
| `PASS` | 0.9643 | - | - | 67 | For how long may ApexCart retain tax and fraud review records? |
| `PASS` | 0.9688 | - | - | 64 | What happens if a customer files a chargeback before contacting support? |
| `PASS` | 1 | - | - | 66 | When can a customer request escalation for an unresolved support issue? |
| `PASS` | 0.5045 | - | - | 96 | Does ApexCart accept cryptocurrency payments for subscriptions? |
| `PASS` | 0.4106 | - | - | 66 | Does ApexCart offer a student discount for annual plans? |

### `benchrun_9a54fb856029` — Hybrid Extractive top-k-5

- Dataset: `ApexCart Policy Reliability Benchmark`
- Status: `completed`
- Cases: `12/12`
- Pass rate: `1`
- Average overall quality: `0.9305`

| Result | Quality | Recall | MRR | Latency | Question |
|---|---:|---:|---:|---:|---|
| `PASS` | 0.9063 | - | - | 86 | How long does a customer have to request a refund for a physical product? |
| `PASS` | 0.8683 | - | - | 96 | When does a monthly subscription cancellation take effect? |
| `PASS` | 0.9555 | - | - | 96 | How many times will ApexCart retry a failed renewal payment? |
| `PASS` | 0.8859 | - | - | 87 | What is the usual delivery time for express shipping? |
| `PASS` | 0.95 | - | - | 96 | What must customers provide when reporting a damaged product? |
| `PASS` | 0.9 | - | - | 91 | What does the hardware accessory warranty exclude? |
| `PASS` | 0.9063 | - | - | 99 | How long does account deletion take after identity verification? |
| `PASS` | 0.9643 | - | - | 98 | For how long may ApexCart retain tax and fraud review records? |
| `PASS` | 0.9688 | - | - | 130 | What happens if a customer files a chargeback before contacting support? |
| `PASS` | 1 | - | - | 155 | When can a customer request escalation for an unresolved support issue? |
| `PASS` | 0.8696 | - | - | 126 | Does ApexCart accept cryptocurrency payments for subscriptions? |
| `PASS` | 0.4764 | - | - | 115 | Does ApexCart offer a student discount for annual plans? |

### `benchrun_29d3bf1d181d` — BM25 Extractive top-k-5

- Dataset: `ApexCart Policy Reliability Benchmark`
- Status: `completed`
- Cases: `12/12`
- Pass rate: `1`
- Average overall quality: `0.9363`

| Result | Quality | Recall | MRR | Latency | Question |
|---|---:|---:|---:|---:|---|
| `PASS` | 0.9063 | - | - | 40 | How long does a customer have to request a refund for a physical product? |
| `PASS` | 0.8983 | - | - | 41 | When does a monthly subscription cancellation take effect? |
| `PASS` | 0.9688 | - | - | 41 | How many times will ApexCart retry a failed renewal payment? |
| `PASS` | 0.9 | - | - | 42 | What is the usual delivery time for express shipping? |
| `PASS` | 0.95 | - | - | 49 | What must customers provide when reporting a damaged product? |
| `PASS` | 0.9 | - | - | 55 | What does the hardware accessory warranty exclude? |
| `PASS` | 0.9063 | - | - | 55 | How long does account deletion take after identity verification? |
| `PASS` | 0.9643 | - | - | 50 | For how long may ApexCart retain tax and fraud review records? |
| `PASS` | 0.9688 | - | - | 49 | What happens if a customer files a chargeback before contacting support? |
| `PASS` | 1 | - | - | 53 | When can a customer request escalation for an unresolved support issue? |
| `PASS` | 0.875 | - | - | 50 | Does ApexCart accept cryptocurrency payments for subscriptions? |
| `PASS` | 0.44 | - | - | 65 | Does ApexCart offer a student discount for annual plans? |

### `benchrun_2e74b4c3f74b` — MiniLM Extractive top-k-5

- Dataset: `ApexCart Policy Reliability Benchmark`
- Status: `completed`
- Cases: `12/12`
- Pass rate: `1`
- Average overall quality: `0.8592`

| Result | Quality | Recall | MRR | Latency | Question |
|---|---:|---:|---:|---:|---|
| `PASS` | 0.8514 | - | - | 8947 | How long does a customer have to request a refund for a physical product? |
| `PASS` | 0.8137 | - | - | 132 | When does a monthly subscription cancellation take effect? |
| `PASS` | 0.8932 | - | - | 111 | How many times will ApexCart retry a failed renewal payment? |
| `PASS` | 0.8291 | - | - | 113 | What is the usual delivery time for express shipping? |
| `PASS` | 0.8646 | - | - | 81 | What must customers provide when reporting a damaged product? |
| `PASS` | 0.8323 | - | - | 81 | What does the hardware accessory warranty exclude? |
| `PASS` | 0.8163 | - | - | 118 | How long does account deletion take after identity verification? |
| `PASS` | 0.8913 | - | - | 77 | For how long may ApexCart retain tax and fraud review records? |
| `PASS` | 0.8796 | - | - | 75 | What happens if a customer files a chargeback before contacting support? |
| `PASS` | 0.9206 | - | - | 90 | When can a customer request escalation for an unresolved support issue? |
| `PASS` | 0.767 | - | - | 112 | Does ApexCart accept cryptocurrency payments for subscriptions? |
| `PASS` | 0.389 | - | - | 101 | Does ApexCart offer a student discount for annual plans? |

### `benchrun_2cbff58687c3` — Hybrid Extractive top-k-5

- Dataset: `ApexCart Policy Reliability Benchmark`
- Status: `completed`
- Cases: `12/12`
- Pass rate: `1`
- Average overall quality: `0.9305`

| Result | Quality | Recall | MRR | Latency | Question |
|---|---:|---:|---:|---:|---|
| `PASS` | 0.9063 | - | - | 93 | How long does a customer have to request a refund for a physical product? |
| `PASS` | 0.8683 | - | - | 89 | When does a monthly subscription cancellation take effect? |
| `PASS` | 0.9555 | - | - | 90 | How many times will ApexCart retry a failed renewal payment? |
| `PASS` | 0.8859 | - | - | 86 | What is the usual delivery time for express shipping? |
| `PASS` | 0.95 | - | - | 76 | What must customers provide when reporting a damaged product? |
| `PASS` | 0.9 | - | - | 82 | What does the hardware accessory warranty exclude? |
| `PASS` | 0.9063 | - | - | 80 | How long does account deletion take after identity verification? |
| `PASS` | 0.9643 | - | - | 78 | For how long may ApexCart retain tax and fraud review records? |
| `PASS` | 0.9688 | - | - | 82 | What happens if a customer files a chargeback before contacting support? |
| `PASS` | 1 | - | - | 103 | When can a customer request escalation for an unresolved support issue? |
| `PASS` | 0.8696 | - | - | 118 | Does ApexCart accept cryptocurrency payments for subscriptions? |
| `PASS` | 0.4764 | - | - | 125 | Does ApexCart offer a student discount for annual plans? |

### `benchrun_95ba9caecb95` — BM25 Extractive top-k-5

- Dataset: `ApexCart Policy Reliability Benchmark`
- Status: `completed`
- Cases: `12/12`
- Pass rate: `1`
- Average overall quality: `0.9363`

| Result | Quality | Recall | MRR | Latency | Question |
|---|---:|---:|---:|---:|---|
| `PASS` | 0.9063 | - | - | 40 | How long does a customer have to request a refund for a physical product? |
| `PASS` | 0.8983 | - | - | 38 | When does a monthly subscription cancellation take effect? |
| `PASS` | 0.9688 | - | - | 44 | How many times will ApexCart retry a failed renewal payment? |
| `PASS` | 0.9 | - | - | 41 | What is the usual delivery time for express shipping? |
| `PASS` | 0.95 | - | - | 55 | What must customers provide when reporting a damaged product? |
| `PASS` | 0.9 | - | - | 45 | What does the hardware accessory warranty exclude? |
| `PASS` | 0.9063 | - | - | 46 | How long does account deletion take after identity verification? |
| `PASS` | 0.9643 | - | - | 41 | For how long may ApexCart retain tax and fraud review records? |
| `PASS` | 0.9688 | - | - | 40 | What happens if a customer files a chargeback before contacting support? |
| `PASS` | 1 | - | - | 50 | When can a customer request escalation for an unresolved support issue? |
| `PASS` | 0.875 | - | - | 54 | Does ApexCart accept cryptocurrency payments for subscriptions? |
| `PASS` | 0.44 | - | - | 56 | Does ApexCart offer a student discount for annual plans? |

### `benchrun_5320b0c1387d` — MiniLM Extractive top-k-5

- Dataset: `ApexCart Policy Reliability Benchmark`
- Status: `completed`
- Cases: `12/12`
- Pass rate: `1`
- Average overall quality: `0.8592`

| Result | Quality | Recall | MRR | Latency | Question |
|---|---:|---:|---:|---:|---|
| `PASS` | 0.8514 | - | - | 9288 | How long does a customer have to request a refund for a physical product? |
| `PASS` | 0.8137 | - | - | 121 | When does a monthly subscription cancellation take effect? |
| `PASS` | 0.8932 | - | - | 104 | How many times will ApexCart retry a failed renewal payment? |
| `PASS` | 0.8291 | - | - | 107 | What is the usual delivery time for express shipping? |
| `PASS` | 0.8646 | - | - | 77 | What must customers provide when reporting a damaged product? |
| `PASS` | 0.8323 | - | - | 78 | What does the hardware accessory warranty exclude? |
| `PASS` | 0.8163 | - | - | 112 | How long does account deletion take after identity verification? |
| `PASS` | 0.8913 | - | - | 81 | For how long may ApexCart retain tax and fraud review records? |
| `PASS` | 0.8796 | - | - | 76 | What happens if a customer files a chargeback before contacting support? |
| `PASS` | 0.9206 | - | - | 70 | When can a customer request escalation for an unresolved support issue? |
| `PASS` | 0.767 | - | - | 111 | Does ApexCart accept cryptocurrency payments for subscriptions? |
| `PASS` | 0.389 | - | - | 70 | Does ApexCart offer a student discount for annual plans? |
