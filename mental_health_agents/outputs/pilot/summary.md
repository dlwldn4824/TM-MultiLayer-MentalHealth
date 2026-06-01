# CounselChat Pilot Benchmark Summary

Generated: 2026-05-29T18:18:23.918421+00:00

## Setup
- Dataset: `nbertagnolli/counsel-chat` (n=29)
- Model: `qwen2.5:7b`
- KB sources: nimh_depression, nimh_anxiety, who_depression
- Evaluation: BERTScore + ROUGE-L (reference), LLM Judge, Retrieval Judge
- No pseudo-labels, no rule-based metrics

## Results by structure

| Structure | BERTScore F1 | ROUGE-L | Faithfulness | Relevancy | Empathy | Safety | Retrieval Rel. | Avg time (s) |
|-----------|--------------|---------|--------------|-----------|---------|--------|----------------|--------------|
| single | 0.764 | 0.132 | 3.66 | 4.59 | 4.21 | 4.62 | 0.00 | 14.2 |
| single_rag | 0.759 | 0.141 | 4.00 | 4.90 | 3.79 | 4.66 | 2.41 | 35.2 |
| two_agent | 0.763 | 0.143 | 3.48 | 4.48 | 3.59 | 4.21 | 2.41 | 46.3 |
| three_agent | 0.763 | 0.143 | 4.03 | 4.86 | 4.31 | 4.83 | 2.41 | 71.5 |

## Interpretation guide

- **BERTScore / ROUGE-L**: lexical-semantic overlap with CounselChat therapist `answerText` (reference).
- **LLM Judge (0–5)**: faithfulness to reference tone/content, relevancy to client question, empathy, safety.
- **Retrieval relevance**: LLM-rated usefulness of official KB chunks (RAG structures only).
- Higher agent count → more LLM calls → longer runtime; compare trade-offs, not only quality.
