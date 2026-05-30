# Multi-Agent Mental Health Reasoning Assessment Framework

로컬 LLM(Ollama) 기반으로 **Single LLM** vs **Multi-Agent + RAG** 구조를 비교하는 연구용 최소 실험 프레임워크입니다.

> **면책:** 본 시스템은 실제 진단·치료·상담 목적이 아닌 **연구용 추론·평가**를 위한 것입니다.

## 프로젝트 목적

정신건강 관련 사용자 발화에 대해 다음 4가지 조건의 추론 품질을 비교합니다.

| 실험 ID | 설명 |
|---------|------|
| `single_llm` | 단일 LLM이 증상·위험·응답을 한 번에 생성 |
| `single_llm_rag` | 단일 LLM + ChromaDB RAG 근거 |
| `multi_agent` | Symptom → Risk → Safety → Consensus 파이프라인 |
| `multi_agent_rag` | Multi-Agent + 각 단계에 RAG 근거 제공 |

비교 지표: 증상 추출(P/R/F1), 위험 분류(accuracy, macro-F1, high-risk recall), 안전성(unsafe/hallucination rate).

## 설치

```bash
cd mental_health_agents
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ollama 실행

1. [Ollama](https://ollama.com) 설치 후 서버 실행:

```bash
ollama serve
```

2. 모델 다운로드 (기본: `qwen2.5:7b`):

```bash
ollama pull qwen2.5:7b
```

3. `config.yaml`에서 `ollama.model`을 변경하면 다른 로컬 모델을 사용할 수 있습니다.

```yaml
ollama:
  model: "qwen2.5:7b"
```

## 데이터셋

- **1차:** [ShenLab/MentalChat16K](https://huggingface.co/datasets/ShenLab/MentalChat16K) (HuggingFace `datasets`)
- `input` 필드에서 사용자/환자 발화를 case text로 사용
- 다운로드 실패 시 `data/sample_cases.csv`로 자동 fallback
- `config.yaml`의 `data.sample_size`로 샘플 수 조정 (기본 100)

### Pseudo Label (Weak Label)

정답 라벨이 없을 때 키워드 규칙으로 `risk_label`, `symptom_labels`를 생성합니다. **임상적 ground truth가 아닙니다.**

## 실험 구조

```
Case Text
    ├─ single_llm ─────────────────────────► JSON output
    ├─ single_llm + RAG ───────────────────► JSON output
    ├─ SymptomAgent → RiskAgent → SafetyAgent → ConsensusAgent
    └─ Multi-Agent + RAG (각 단계에 evidence)
```

RAG: **`sources.yaml`에 고정된 공식 URL만** `scripts/fetch_official_knowledge.py`로 HTML 수집 → `official/*.md` 저장 → ChromaDB 인덱싱. DSM/ICD 원문·유료 매뉴얼은 fetch 금지(화이트리스트 + deny 패턴).

```bash
python scripts/fetch_official_knowledge.py   # NIMH/WHO/NICE 공개 페이지만
python scripts/build_knowledge_db.py
```

`knowledge_base.mode`: `official` | `mock` | `both`

## Sequential Experimental Design (단계적 확장)

**Exhaustive grid search가 아니라**, Phase별 실행 후 **상위 1~2개 후보만 다음 Phase로 전달(pruning)** 합니다.

| Phase | 목적 | 기본 설정 |
|-------|------|-----------|
| 0 | 파이프라인 smoke test | sample CSV 10, Single LLM |
| 1 | 구조 비교 (4종) | MentalChat16K 100 |
| 2 | 모델 비교 | Phase1 최고 구조 × `models` 목록 |
| 3 | 앙상블 | majority / equal / performance / safety weight |
| 4 | 가중치 grid | `weight_grid` (합=1) |
| 5 | Agent/Layer ablation | `ablation_targets` |
| 6 | 데이터 스케일링 | `sample_sizes`: 100 / 300 / 1000 |

**평가 우선순위:** `high_risk_recall` ↑ → `unsafe_flag_rate` ↓ → `symptom_f1` ↑

`config.yaml`의 `experiment_phases`에서 Phase별 on/off 가능합니다.

## 실행 명령어

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b
python scripts/build_knowledge_db.py
python scripts/run_phases.py
```

### Phase만 실행

```bash
python scripts/run_phases.py --only-phase phase0_smoke_test
python scripts/run_phases.py --only-phase phase1_structure_comparison
```

### 레거시: 4구조 일괄 실행 (pruning 없음)

```bash
python scripts/run_experiments.py --sample-size 100
```

### Smoke test (Phase 0와 동일)

```bash
python scripts/run_phases.py --only-phase phase0_smoke_test
```

## 결과 파일

| 경로 | 설명 |
|------|------|
| `outputs/results.sqlite` | cases, **experiment_runs**, agent_outputs, predictions, metrics |
| `outputs/predictions.csv` | case × experiment (model, phase, structure 등 메타 포함) |
| `outputs/metrics.json` | 실험별 평가 지표 |
| `outputs/metrics.csv` | 모든 run 요약 표 (보고서용) |
| `outputs/experiment_summary.md` | 최고 조합 자동 요약 |
| `outputs/phase_state.json` | Phase 간 pruning 상태 |
| `outputs/chroma_db/` | RAG 벡터 인덱스 |

`experiment_runs` 테이블에는 model, sample_size, use_rag, structure, ablation, ensemble_method, weights, json_parse_success_rate, avg_runtime_sec가 저장됩니다.

## Training Expansion Framework

Inference 실험 결과를 학습 데이터로 변환하는 **2단계 연구 파이프라인**입니다.

| 단계 | 설명 |
|------|------|
| **Inference-only** | `run_phases.py` — 구조/모델/앙상블 비교 |
| **Learning** | `dataset_builder` → `training_runner` → `evaluation_compare` |

### 학습 방식 차이

| 방식 | 목적 | 산출물 |
|------|------|--------|
| **SFT (LoRA)** | Multi-Agent 출력·정답 JSON 모방 | `training/sft_dataset/` |
| **DPO** | 안전·정확 응답 vs 실패 응답 preference | `training/dpo_dataset/dpo_pairs.jsonl` |
| **Distillation** | Teacher(Multi-Agent+RAG) → Student(Single) | `training/distillation_dataset/` |

### Sequential Research Workflow

```text
Stage 1  Inference Experiment     (run_phases.py)
    ↓
Stage 2  Failure Collection      (failure_analysis.py)
    ↓
Stage 3  Dataset Generation       (dataset_builder.py)
    ↓
Stage 4  LoRA Training             (training_runner.py --mode lora)
    ↓
Stage 5  DPO Training              (training_runner.py --mode dpo)
    ↓
Stage 6  Distillation              (training_runner.py --mode distill)
    ↓
Stage 7  Final Evaluation          (evaluation_compare.py)
```

### Training CLI

```bash
# 1) 데이터셋만 (smoke)
python src/dataset_builder.py --smoke

# 2) LoRA / DPO / Distill (smoke → mini → full)
pip install -r requirements-training.txt
python src/training_runner.py --model qwen2.5:7b --mode lora --scale smoke

# 3) 전체 smoke 파이프라인
python scripts/run_training_pipeline.py --stage all_smoke

# 4) Base vs fine-tuned 비교
python src/evaluation_compare.py --smoke
```

### Training 산출물

| 경로 | 내용 |
|------|------|
| `training/failure_dataset/failure_cases.jsonl` | hallucination, risk mismatch 등 |
| `training/sft_dataset/sft_flat.jsonl` | 단일 JSON SFT |
| `training/sft_dataset/sft_chain.jsonl` | 단계별 chain SFT |
| `training/dpo_dataset/dpo_pairs.jsonl` | chosen / rejected |
| `training/distillation_dataset/teacher_pairs.jsonl` | teacher trace |
| `training/checkpoints/` | LoRA / DPO / distill 체크포인트 |
| `training/reports/training_summary.md` | 모델 비교 요약 |

### 연구 기여 (보고서용)

- Failure-driven dataset curation from agent traces
- Multi-format SFT (Alpaca / ChatML / task-specific)
- Preference optimization for safety & grounding
- Multi-agent → single-model distillation
- Stage-wise evaluation (not end-to-end accuracy only)

## 모듈 구조

```
src/
  dataset_builder.py       # Stage 2–3 orchestration
  failure_analysis.py      # Phase A failure JSONL
  sft_builder.py           # Phase B/C SFT
  dpo_builder.py           # Phase E DPO pairs
  distillation_builder.py  # Phase G teacher→student
  training_runner.py       # Phase D/F LoRA & DPO train
  evaluation_compare.py    # Phase H model comparison
  phases.py          # Phase 0–6 pipeline + pruning
  run_config.py      # RunConfig (model/structure/phase metadata)
  pruning.py         # 후보 순위·선택
  ensemble.py        # weighted voting 앙상블
  experiments.py     # 단일 run 실행
  evaluation.py      # metrics + summary
  agents.py          # 4 agents + ablation flags
  ...
scripts/
  run_phases.py      # Sequential Experimental Design 진입점
  run_experiments.py # 레거시 일괄 실행
```

## 연구 한계

- **Pseudo label** 기반이므로 clinical ground truth가 아닙니다.
- **로컬 LLM** 성능은 모델 크기·양자화·하드웨어에 크게 의존합니다.
- 본 시스템은 **실제 진단/상담용이 아니라** 연구용 평가 프레임워크입니다.
- MentalChat16K는 **상담 응답 데이터** 성격이 강해 risk classification 벤치마크로는 한계가 있습니다.
- RAG 지식 문서는 요약·가이드라인 수준이며 의료기관 공식 프로토콜을 대체하지 않습니다.

## 라이선스 / 윤리

연구 목적으로만 사용하세요. 위기 상황에서는 반드시 현지 전문 기관·응급 서비스에 연락해야 합니다.
