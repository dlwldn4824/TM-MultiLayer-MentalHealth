# CounselChat Pilot Benchmark

Fast reference-based comparison: **Single vs Single+RAG vs 2-Agent vs 3-Agent**.

- Dataset: [nbertagnolli/counsel-chat](https://huggingface.co/datasets/nbertagnolli/counsel-chat) (HF `load_dataset`)
- Reference: therapist `answerText` (no pseudo-labels, no rule-based metrics)
- KB: NIMH Depression, NIMH Anxiety, WHO Depression (55 chunks)

## Run

```bash
cd mental_health_agents

# First time on a new machine (creates .venv + installs requirements)
./scripts/setup_venv.sh

# `python scripts/run_pilot.py` auto-switches to .venv when present
source .venv/bin/activate   # optional

# Recommended: resume is ON by default; each case saved immediately to predictions.csv
./scripts/run_pilot_resume.sh

# Or manually (same behavior)
python scripts/run_pilot.py --no-rebuild-kb

# After interrupt — run again (skips finished case×structure pairs)
python scripts/run_pilot.py --no-rebuild-kb

# Start completely over
python scripts/run_pilot.py --fresh --no-rebuild-kb

# Check progress
cat outputs/pilot/progress.json

# Evaluation only (when predictions.csv has 120 rows)
python scripts/run_pilot.py --eval-only

# Reference metrics only (skip LLM judges)
python scripts/run_pilot.py --eval-only --skip-llm-judge --skip-retrieval-judge
# or after predictions exist:
python scripts/run_pilot.py --eval-only
```

## Outputs (`outputs/pilot/`)

| File | Description |
|------|-------------|
| `predictions.csv` | Per case × structure responses |
| `metrics.json` | Aggregated BERTScore, ROUGE-L, judge means |
| `judge_scores.csv` | Per-row LLM judge (0–5) |
| `retrieval_scores.csv` | Per-row retrieval judge (RAG structures) |
| `summary.md` | Comparison table |

## Structures

1. **single** — one-shot counseling reply  
2. **single_rag** — reply with Chroma retrieval  
3. **two_agent** — RetrieverAgent → ReasoningAgent  
4. **three_agent** — RetrieverAgent → ReasoningAgent → SafetyAgent  

Config: `config_pilot.yaml`
