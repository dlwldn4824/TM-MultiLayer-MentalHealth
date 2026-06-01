# Experiment Summary

Generated: 2026-05-29T07:31:52.675009+00:00

## Sequential Experimental Design

This project uses phased expansion with pruning (not exhaustive grid search).

## Phase State

```json
{
  "phase0_passed": true,
  "phase0_metrics": {
    "symptom_precision": 0.5555555555555556,
    "symptom_recall": 0.5,
    "symptom_f1": 0.5263157894736842,
    "risk_accuracy": 0.3,
    "risk_macro_f1": 0.35555555555555557,
    "high_risk_recall": 1.0,
    "unsafe_flag_rate": 0.4,
    "hallucination_flag_rate": 0.0,
    "avg_runtime_sec": 23.972118545812556,
    "total_runtime_sec": 239.72118545812555,
    "json_parse_success_rate": 1.0
  }
}
```

## Best Overall Run

- **Experiment:** `phase0__single_llm__qwen2_5_7b__n10__smoke`
- **Phase:** phase0
- **Model:** qwen2.5:7b
- **Structure:** single_llm
- **Sample size:** 10
- **High-risk recall:** 1.000
- **Unsafe rate:** 0.400
- **Symptom F1:** 0.526
- **JSON success:** 1.000

## All Runs (ranked)

| Rank | Experiment | Phase | Model | Structure | High-Risk Recall | Unsafe ↓ | Symptom F1 | JSON | Time(s) |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `phase0__single_llm__qwen2_5_7b__n10__smo` | phase0 | qwen2.5:7b | single_llm | 1.000 | 0.400 | 0.526 | 1.000 | 24.0 |