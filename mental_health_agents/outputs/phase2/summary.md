# Phase 2: Model Comparison Benchmark Summary

Generated: 2026-05-30T13:56:14.958651+00:00

## Setup
- Dataset: `nbertagnolli/counsel-chat` (n=30)
- Structures: single_rag, three_agent
- Models: qwen2.5:7b, llama3.1:8b, gemma2:9b, mistral:7b, qwen2.5:14b
- Evaluation: BERTScore F1, ROUGE-L, LLM Judge (4 criteria), Retrieval Relevance
- Runtime: avg seconds, token throughput, inference stability, JSON success rate

## Table 1 — Structure × Model Quality Matrix

| Model | Structure | BERTScore | ROUGE-L | Faith | Relev | Empathy | Safety | Retrieval | Runtime(s) | JSON OK |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma2:9b | single_rag | 0.766 | 0.142 | 3.69 | 4.69 | 3.86 | 4.52 | 2.41 | 25.9 | 97% |
| gemma2:9b | three_agent | 0.767 | 0.147 | 3.69 | 4.66 | 4.17 | 4.83 | 2.41 | 71.7 | 100% |
| llama3.1:8b | single_rag | 0.767 | 0.149 | 3.41 | 4.34 | 3.86 | 4.10 | 2.41 | 24.0 | 86% |
| llama3.1:8b | three_agent | 0.756 | 0.144 | 3.66 | 4.62 | 4.24 | 4.76 | 2.41 | 56.9 | 93% |
| mistral:7b | single_rag | 0.750 | 0.140 | 3.24 | 4.28 | 3.31 | 4.31 | 2.41 | 32.5 | 100% |
| mistral:7b | three_agent | 0.755 | 0.139 | 3.34 | 4.17 | 4.00 | 4.00 | 2.41 | 76.9 | 97% |
| qwen2.5:14b | single_rag | 0.766 | 0.139 | 3.69 | 4.66 | 3.62 | 4.76 | 2.41 | 77.6 | 100% |
| qwen2.5:14b | three_agent | 0.765 | 0.133 | 3.76 | 4.79 | 4.03 | 4.83 | 2.41 | 221.2 | 100% |
| qwen2.5:7b | single_rag | 0.760 | 0.142 | 3.90 | 4.86 | 3.79 | 4.86 | 2.41 | 34.0 | 10% |
| qwen2.5:7b | three_agent | 0.760 | 0.139 | 4.00 | 4.86 | 4.17 | 4.93 | 2.41 | 72.5 | 70% |

## Table 2 — Best Model Ranking

| Rank | Model | Structure | Score | Faith | Safety | Empathy | Relev | Runtime(s) |
|------|-------|-----------|-------|-------|--------|---------|-------|------------|
| 1 | qwen2.5:7b | three_agent | 69.40 | 4.00 | 4.93 | 4.17 | 4.86 | 72.5 |
| 2 | qwen2.5:7b | single_rag | 69.40 | 3.90 | 4.86 | 3.79 | 4.86 | 34.0 |
| 3 | llama3.1:8b | three_agent | 67.49 | 3.66 | 4.76 | 4.24 | 4.62 | 56.9 |
| 4 | gemma2:9b | single_rag | 67.25 | 3.69 | 4.52 | 3.86 | 4.69 | 25.9 |
| 5 | gemma2:9b | three_agent | 67.06 | 3.69 | 4.83 | 4.17 | 4.66 | 71.7 |
| 6 | qwen2.5:14b | single_rag | 64.84 | 3.69 | 4.76 | 3.62 | 4.66 | 77.6 |
| 7 | llama3.1:8b | single_rag | 63.62 | 3.41 | 4.10 | 3.86 | 4.34 | 24.0 |
| 8 | mistral:7b | single_rag | 61.37 | 3.24 | 4.31 | 3.31 | 4.28 | 32.5 |
| 9 | mistral:7b | three_agent | 60.28 | 3.34 | 4.00 | 4.00 | 4.17 | 76.9 |
| 10 | qwen2.5:14b | three_agent | 59.79 | 3.76 | 4.83 | 4.03 | 4.79 | 221.2 |

## Analysis

### 1. Is Three-Agent consistently better across models?
Three-Agent leads on faithfulness/empathy/safety vs single_rag in 4/5 models tested.

### 2. Does qwen2.5:14b actually outperform qwen2.5:7b?
14B aggregate faith+safety mean=8.52 vs 7B=8.84. 7B matches or beats 14B on quality; 7B is faster.

### 3. Do smaller models (mistral/gemma) preserve Multi-Agent gains?
Multi-Agent faithfulness delta vs single_rag: mistral:7b: Δfaith=+0.10, gemma2:9b: Δfaith=+0.00

### 4. What is the quality–latency trade-off?
Fastest avg runtime: llama3.1:8b + single_rag (24.0s). Highest faithfulness: qwen2.5:7b + three_agent (4.00/5).

### 5. Which model–structure pair is the best overall candidate?
Best overall (weighted rank): qwen2.5:7b + three_agent (score=69.40).
