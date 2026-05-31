# Phase 3: Ensemble Experiment

Phase 3 compares **ensemble inference strategies** using Phase 2 top model candidates on CounselChat.

## Prerequisites

Phase 2 must be complete (predictions + metrics + judge scores):

```bash
cd mental_health_agents
python scripts/run_phase2.py --sample-size 30
# or eval-only if predictions exist:
python scripts/run_phase2.py --eval-only
```

## Ensemble Methods

| Method | Description |
|--------|-------------|
| `majority_selection` | Per-case judge dimension voting; pick response with most wins |
| `equal_weight` | Equal-weight average across Faithfulness / Relevancy / Empathy / Safety |
| `judge_weighted` | Weighted by judge criteria + Phase 2 model performance |
| `safety_prioritized` | Safety-heavy weighting for mental-health robustness |

## Run

```bash
# Smoke (build ensembles from Phase 2, skip re-evaluation)
python scripts/run_phase3.py --skip-eval

# Full evaluation (BERTScore, ROUGE-L, LLM Judge, Runtime)
python scripts/run_phase3.py

# Fast metrics (skip LLM judge)
python scripts/run_phase3.py --skip-llm-judge

# Override candidates
python scripts/run_phase3.py --models qwen2.5:7b,llama3.1:8b,gemma2:9b --structure single_rag
```

## Outputs (`outputs/phase3/`)

| File | Description |
|------|-------------|
| `ensemble_results.csv` | Ranked ensemble method comparison |
| `ensemble_summary.md` | Tables + analysis questions |
| `metrics.json` / `metrics.csv` | Per-method aggregates |
| `predictions.csv` | Per-case ensemble responses |
| `judge_scores.csv` | Per-case LLM judge scores |
| `runtime_metrics.csv` | Runtime per ensemble method |

## Analysis Questions

1. Is ensemble consistently better than single best model?
2. Can faithfulness and safety improve together?
3. Is runtime increase acceptable vs single-model inference?

## Config

Edit `config_phase3.yaml` for Phase 2 input paths, `top_k_models`, and ensemble methods.
