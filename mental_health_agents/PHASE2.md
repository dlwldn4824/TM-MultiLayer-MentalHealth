# Phase 2: Model Comparison Benchmark

Phase 1 (pilot) compared **structures** with one model. Phase 2 compares **models** on the two Phase 1 finalists:

- `single_rag` — high answer relevancy
- `three_agent` — best faithfulness / empathy / safety

Reuses CounselChat dataset, official KB, BERTScore, ROUGE-L, LLM Judge, and Retrieval Judge from `src/pilot/`.

## Setup

```bash
cd mental_health_agents
./scripts/setup_venv.sh
ollama pull qwen2.5:7b   # pull models you plan to run
```

## Run (staged default order)

```bash
# Full grid: 5 models × 2 structures × 30 cases
python scripts/run_phase2.py

# Subset / smoke
python scripts/run_phase2.py --sample-size 5 --models qwen2.5:7b --structures single_rag

# Multiple models
python scripts/run_phase2.py --models qwen2.5:7b,llama3.1:8b

# Eval only (predictions.csv complete)
python scripts/run_phase2.py --eval-only

# Fast metrics (skip judges)
python scripts/run_phase2.py --eval-only --skip-llm-judge --skip-retrieval-judge
```

Resume is **on** by default (`outputs/phase2/predictions.csv` keyed by `case_id + model + structure`).

## Outputs (`outputs/phase2/`)

| File | Description |
|------|-------------|
| `metrics.json` | Per model×structure aggregates |
| `metrics.csv` | Same as flat table |
| `model_comparison_matrix.csv` | Table 1 export |
| `summary.md` | Tables + auto analysis |
| `judge_scores.csv` | Per-case LLM judge |
| `retrieval_scores.csv` | Per-case retrieval judge |
| `runtime_metrics.csv` | Runtime / throughput / stability |
| `predictions.csv` | Checkpointed inference rows |

SQLite: `outputs/results.sqlite` → table `phase2_counsel_benchmark`.

## Config

Edit `config_phase2.yaml` for models, structures, sample size, evaluation settings.

Phase 1 (`config_pilot.yaml`, `scripts/run_pilot.py`) is unchanged.
