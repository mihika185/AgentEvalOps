# AgentEvalOps Benchmark Report

Generated at: `2026-07-18T03:41:29.498137+00:00`

## Scope

- Dataset filter: `dataset_c0b9622b9072`
- Experiment filter: `not included`
- Benchmark runs included: `4`

## Benchmark Run Summary

| Pipeline | Provider / Model | Cases | Answerable | Unanswerable | Pass Rate | Avg Quality | Avg Hallucination | Avg Latency | Avg Cost | Recall@k | MRR | nDCG@k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hybrid Retrieval + Rerank | `groq:llama-3.3-70b-versatile` | 12/12 | 10/10 | 2/2 | 1 | 0.9735 | 0.0067 | 726.75 ms | $0.0003 | 1 | 1 | 1 |
| Hybrid Retrieval | `groq:llama-3.3-70b-versatile` | 12/12 | 10/10 | 2/2 | 1 | 0.9731 | 0.0067 | 2446.3333 ms | $0.0005 | 1 | 1 | 1 |
| Dense Retrieval | `groq:llama-3.3-70b-versatile` | 11/12 | 9/10 | 2/2 | 0.9167 | 0.8522 | 0.1067 | 3106.25 ms | $0.0005 | 0.9 | 0.9 | 0.9 |
| BM25 Baseline | `groq:llama-3.3-70b-versatile` | 12/12 | 10/10 | 2/2 | 1 | 0.9732 | 0 | 431.8333 ms | $0.0005 | 1 | 1 | 1 |

## Best Benchmark Run

- Benchmark run: `benchrun_17caf2f6c17a`
- Dataset: `ApexCart Policy Reliability Benchmark`
- Pipeline: Hybrid Retrieval + Rerank
- Answer generator: `groq:llama-3.3-70b-versatile`
- Passed cases: `12/12`
- Answerable cases: `10/10`
- Unanswerable cases: `2/2`
- Pass rate: `1`
- Average answer support: `0.9929`
- Average query-answer relevance: `0.9027`
- Average hallucination risk: `0.0067`
- Average overall quality: `0.9735`
- Average latency: `726.75 ms`
- Average estimated cost: `$0.0003`
- Total estimated cost: `$0.004`
- Average total tokens: `551.8333`
- Average Recall@k: `1`
- Average Precision@k: `0.3666`
- Average MRR: `1`
- Average nDCG@k: `1`

Selection rule: highest pass rate, then highest average overall quality.

## Benchmark Run Details

### `benchrun_17caf2f6c17a` — Hybrid Retrieval + Rerank

- Dataset: `ApexCart Policy Reliability Benchmark`
- Status: `completed`
- Cases: `12/12`
- Pass rate: `1`
- Average overall quality: `0.9735`

| Result | Quality | Recall | MRR | Latency | Question |
|---|---:|---:|---:|---:|---|
| `PASS` | 0.9714 | 1 | 1 | 3102 | How long does a customer have to request a refund for a physical product? |
| `PASS` | 0.9583 | 1 | 1 | 504 | When does a monthly subscription cancellation take effect? |
| `PASS` | 1 | 1 | 1 | 595 | How many times will ApexCart retry a failed renewal payment? |
| `PASS` | 0.95 | 1 | 1 | 478 | What is the usual delivery time for express shipping? |
| `PASS` | 0.9786 | 1 | 1 | 538 | What must customers provide when reporting a damaged product? |
| `PASS` | 0.925 | 1 | 1 | 621 | What does the hardware accessory warranty exclude? |
| `PASS` | 0.9833 | 1 | 1 | 582 | How long does account deletion take after identity verification? |
| `PASS` | 1 | 1 | 1 | 485 | For how long may ApexCart retain tax and fraud review records? |
| `PASS` | 0.9688 | 1 | 1 | 425 | What happens if a customer files a chargeback before contacting support? |
| `PASS` | 1 | 1 | 1 | 549 | When can a customer request escalation for an unresolved support issue? |
| `PASS` | 0.31 | - | - | 420 | Does ApexCart accept cryptocurrency payments for subscriptions? |
| `PASS` | 0.31 | - | - | 422 | Does ApexCart offer a student discount for annual plans? |

### `benchrun_485f9cc1e267` — Hybrid Retrieval

- Dataset: `ApexCart Policy Reliability Benchmark`
- Status: `completed`
- Cases: `12/12`
- Pass rate: `1`
- Average overall quality: `0.9731`

| Result | Quality | Recall | MRR | Latency | Question |
|---|---:|---:|---:|---:|---|
| `PASS` | 0.9714 | 1 | 1 | 4842 | How long does a customer have to request a refund for a physical product? |
| `PASS` | 0.9583 | 1 | 1 | 3624 | When does a monthly subscription cancellation take effect? |
| `PASS` | 0.9838 | 1 | 1 | 3771 | How many times will ApexCart retry a failed renewal payment? |
| `PASS` | 0.9363 | 1 | 1 | 4021 | What is the usual delivery time for express shipping? |
| `PASS` | 0.9786 | 1 | 1 | 3681 | What must customers provide when reporting a damaged product? |
| `PASS` | 0.95 | 1 | 1 | 3631 | What does the hardware accessory warranty exclude? |
| `PASS` | 0.9833 | 1 | 1 | 3784 | How long does account deletion take after identity verification? |
| `PASS` | 1 | 1 | 1 | 524 | For how long may ApexCart retain tax and fraud review records? |
| `PASS` | 0.9688 | 1 | 1 | 322 | What happens if a customer files a chargeback before contacting support? |
| `PASS` | 1 | 1 | 1 | 390 | When can a customer request escalation for an unresolved support issue? |
| `PASS` | 0.3659 | - | - | 424 | Does ApexCart accept cryptocurrency payments for subscriptions? |
| `PASS` | 0.2954 | - | - | 342 | Does ApexCart offer a student discount for annual plans? |

### `benchrun_d9c217bd6a2c` — Dense Retrieval

- Dataset: `ApexCart Policy Reliability Benchmark`
- Status: `completed`
- Cases: `11/12`
- Pass rate: `0.9167`
- Average overall quality: `0.8522`

| Result | Quality | Recall | MRR | Latency | Question |
|---|---:|---:|---:|---:|---|
| `PASS` | 0.9303 | 1 | 1 | 10237 | How long does a customer have to request a refund for a physical product? |
| `PASS` | 0.8948 | 1 | 1 | 499 | When does a monthly subscription cancellation take effect? |
| `FAIL` | 0.2833 | 0 | 0 | 483 | How many times will ApexCart retry a failed renewal payment? |
| `PASS` | 0.8968 | 1 | 1 | 423 | What is the usual delivery time for express shipping? |
| `PASS` | 0.9145 | 1 | 1 | 472 | What must customers provide when reporting a damaged product? |
| `PASS` | 0.8993 | 1 | 1 | 2659 | What does the hardware accessory warranty exclude? |
| `PASS` | 0.9159 | 1 | 1 | 4835 | How long does account deletion take after identity verification? |
| `PASS` | 0.9452 | 1 | 1 | 4055 | For how long may ApexCart retain tax and fraud review records? |
| `PASS` | 0.9019 | 1 | 1 | 3654 | What happens if a customer files a chargeback before contacting support? |
| `PASS` | 0.9405 | 1 | 1 | 2661 | When can a customer request escalation for an unresolved support issue? |
| `PASS` | 0.289 | - | - | 3717 | Does ApexCart accept cryptocurrency payments for subscriptions? |
| `PASS` | 0.2492 | - | - | 3580 | Does ApexCart offer a student discount for annual plans? |

**Failures**

- `benchitem_855562b06a48`: Expected answerable response, but quality gate blocked the answer

### `benchrun_908e3a60c33d` — BM25 Baseline

- Dataset: `ApexCart Policy Reliability Benchmark`
- Status: `completed`
- Cases: `12/12`
- Pass rate: `1`
- Average overall quality: `0.9732`

| Result | Quality | Recall | MRR | Latency | Question |
|---|---:|---:|---:|---:|---|
| `PASS` | 0.9714 | 1 | 1 | 746 | How long does a customer have to request a refund for a physical product? |
| `PASS` | 0.9583 | 1 | 1 | 341 | When does a monthly subscription cancellation take effect? |
| `PASS` | 1 | 1 | 1 | 314 | How many times will ApexCart retry a failed renewal payment? |
| `PASS` | 0.9 | 1 | 1 | 519 | What is the usual delivery time for express shipping? |
| `PASS` | 1 | 1 | 1 | 388 | What must customers provide when reporting a damaged product? |
| `PASS` | 0.95 | 1 | 1 | 460 | What does the hardware accessory warranty exclude? |
| `PASS` | 0.9833 | 1 | 1 | 431 | How long does account deletion take after identity verification? |
| `PASS` | 1 | 1 | 1 | 446 | For how long may ApexCart retain tax and fraud review records? |
| `PASS` | 0.9688 | 1 | 1 | 479 | What happens if a customer files a chargeback before contacting support? |
| `PASS` | 1 | 1 | 1 | 436 | When can a customer request escalation for an unresolved support issue? |
| `PASS` | 0.34 | - | - | 313 | Does ApexCart accept cryptocurrency payments for subscriptions? |
| `PASS` | 0.2767 | - | - | 309 | Does ApexCart offer a student discount for annual plans? |
