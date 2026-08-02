# Conditional Bidirectional Experiment

Generated: 2026-06-19T03:26:54.282413+00:00

## Pipeline

```
Question → Retrieval → Response → Safety Gatekeeper
  → (if issues) Revision → Safety Recheck → Final Answer
  → (if pass) Final Answer = Response
```

## vs Phase 2 prototype (`three_agent`)

| | three_agent (prototype) | conditional_bidirectional |
|--|-------------------------|---------------------------|
| Retrieval | RetrieverAgent + RAG | RAG only |
| Response | ReasoningAgent | single_rag-style Response |
| Safety | SafetyAgent (always revise-capable) | Gatekeeper → conditional Revision → Recheck |
| LLM calls (pass path) | 3 | 3 |
| LLM calls (revision path) | 3 | 5 |

## Run stats (conditional_bidirectional)

- Cases: **30**
- Gatekeeper pass (no revision): **96.7%**
- Revision triggered: **3.3%**
- Avg LLM calls: **3.07**
- Avg runtime: **45.6s**

- three_agent avg runtime (Phase 2): **72.5s**
- single_rag avg runtime (Phase 2): **34.0s**

## Judge comparison (this run)

### conditional_bidirectional

- faithfulness: **3.97**
- answer_relevancy: **4.90**
- empathy: **3.87**
- safety: **4.70**
