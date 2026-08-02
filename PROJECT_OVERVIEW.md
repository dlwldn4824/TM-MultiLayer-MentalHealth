# TM-MultiLayer-MentalHealth — 프로젝트 전체 설명

정신건강 상담 도메인에서 **Single LLM**과 **Multi-Agent(+RAG)**, 그리고 **Conditional Bidirectional(조건부 Revision)** 구조를 비교·분석하고, 실험 로그를 학습 데이터로 확장(LoRA/DPO/Distillation)할 수 있게 만든 **연구용 실험 프레임워크**입니다.

> **면책:** 본 프로젝트는 임상 진단·치료·상담 목적이 아닌 **연구/평가용**입니다. 실제 서비스 적용 시 임상·법적 검토가 필요합니다.

---

## 프로젝트 목표

- **구조 관점 비교**: 단일 모델 응답 vs 에이전트 분업(증상/위험/안전/합의) 파이프라인의 품질·안전성·런타임 trade-off 분석
- **RAG 관점 비교**: 공식 출처(NIMH/WHO/NICE) 기반 근거 검색이 응답에 미치는 영향 분석
- **조건부 안전성 강화**: “항상 multi-agent”가 아니라, **Gatekeeper가 문제를 탐지한 경우에만 Revision**하는 Conditional 구조 실험
- **학습 확장(옵션)**: inference trace로 SFT/DPO/Distillation 데이터셋을 생성하고 튜닝 실험으로 연결

---

## 핵심 구성요소 (Architecture)

### 1) RAG (공식 출처 화이트리스트)

- 공식 문서(NIMH/WHO/NICE)만 수집 → markdown 저장 → chunking → ChromaDB 인덱싱
- 목표: hallucination 완화, 근거 기반 답변(grounding)

관련 스크립트:
- `mental_health_agents/scripts/fetch_official_knowledge.py`
- `mental_health_agents/scripts/build_knowledge_db.py`

### 2) Single / Multi-Agent 추론 구조

**MentalChat16K(Phase 1)** 기준 구조(4조건):
- `single_llm`
- `single_llm_rag`
- `multi_agent`
- `multi_agent_rag`

**CounselChat(Phase 2)** 기준 구조(2조건 + 추가 실험):
- `single_rag`: 1회 생성 + RAG evidence block
- `three_agent`: Retriever → Reasoning → Safety (항상 3회 호출)
- `conditional_bidirectional` (추가 실험): Response → Gatekeeper → (필요 시) Revision → Recheck

#### Conditional Bidirectional (실험 파이프라인)

```
Question → Retrieval → Response → Safety Gatekeeper
  → (if issues) Revision → Safety Recheck → Final Answer
  → (if pass) Final Answer = Response
```

의도:
- 위험/부적절 조언 가능성이 있는 경우에만 추가 연산(Revision/Recheck)을 사용해
  **안전성 강화**와 **런타임 비용**을 균형 있게 맞추는 구조

---

## 데이터셋

### Phase 1 (구조 비교)
- **MentalChat16K** (`ShenLab/MentalChat16K`)
- pseudo-label(약한 라벨) 기반 지표(증상/위험/unsafe 등)를 사용 (임상 ground truth 아님)

### Phase 2 (모델 비교)
- **CounselChat** (`nbertagnolli/counsel-chat`) 30 샘플
- reference therapist answer를 사용해 BERTScore/ROUGE/LLM Judge로 평가

---

## 평가 (Evaluation)

### Phase 1 (pseudo-label 기반)
- Symptom P/R/F1
- Risk accuracy/macro-F1/high-risk recall
- unsafe/hallucination flag rate
- runtime

### Phase 2 (reference 기반 + Judge)
- BERTScore F1, ROUGE-L
- LLM Judge (0–5): faithfulness / answer_relevancy / empathy / safety
- retrieval relevance (RAG 구조에 한해)
- runtime + JSON parse success rate

---

## 현재까지 실험 결과 요약

### Phase 1 — 구조 비교 (MentalChat16K 100건)
- 결과 요약 파일: `mental_health_agents/outputs/experiment_results_summary.md`
- 관찰: Multi-Agent+RAG가 symptom/risk 지표에서 유리한 구간이 있으나, pseudo-label 한계·분포 편향(high=0) 등 **평가 설계 한계**가 명시됨

### Phase 2 — 모델 비교 (CounselChat 30건, 5모델 × 2구조)
- 요약 파일: `mental_health_agents/outputs/phase2/summary.md`
- Best quality 후보(Phase 2 기준): **`qwen2.5:7b + three_agent`**
- Best speed 후보: **`llama3.1:8b + single_rag`**
- 관찰: three_agent가 다수 모델에서 empathy/safety를 개선하지만 runtime 증가가 큼

### 추가 실험 — Conditional Bidirectional (CounselChat 동일 세트)
- 요약 파일: `mental_health_agents/outputs/phase2/conditional_experiment/conditional_summary.md`
- 핵심 수치(conditional_bidirectional):
  - Gatekeeper pass(Revision 없음): **96.7%**
  - Revision triggered: **3.3%**
  - Avg runtime: **45.6s** (three_agent 72.5s, single_rag 34.0s 대비 중간)
  - Judge 평균(0–5): faith 3.97 / relev 4.90 / empathy 3.87 / safety 4.70

---

## 실행 방법 (Quickstart)

### 1) 환경 구성

```bash
cd mental_health_agents
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Ollama 실행 + 모델 준비

```bash
ollama serve
ollama pull qwen2.5:7b
```

### 3) KB 인덱스 구축 (RAG)

```bash
python scripts/fetch_official_knowledge.py
python scripts/build_knowledge_db.py
```

### 4) Phase 실행

```bash
python scripts/run_phases.py
```

### 5) Phase 2 (CounselChat 모델 비교)

```bash
python scripts/run_phase2.py
```

### 6) Conditional Bidirectional 실험

```bash
python scripts/run_conditional_experiment.py
```

---

## 산출물 (Outputs)

핵심 산출물 디렉토리: `mental_health_agents/outputs/`

- `predictions.csv`: 케이스별 모델 출력 로그
- `metrics.json` / `metrics.csv`: 집계 지표
- `results.sqlite`: 실험/에이전트/평가 결과 DB
- `experiment_results_summary.md`: Phase 1 결과 해석
- `phase2/summary.md`: Phase 2 모델 비교 요약
- `phase2/revision_analysis/`: Revision(초안→수정) 분석(실험 1~3)
- `phase2/conditional_experiment/`: Conditional Bidirectional 재실험 산출물

---

## 디렉토리 구조 (요약)

```
multilayer_tuning_mental_heath_agent/
  mental_health_agents/
    src/                 # core pipeline (agents, rag, evaluation, phases)
    scripts/             # CLI runners (phases, phase2, conditional experiment, KB build)
    data/knowledge/      # official KB (NIMH/WHO/NICE) + metadata
    outputs/             # predictions/metrics/sqlite/summaries
    training/            # (옵션) SFT/DPO/Distillation datasets & checkpoints
  TEAM_EXECUTION_GUIDE.md
  PROJECT_OVERVIEW.md
```

---

## 한계 및 주의사항

- **임상적 ground truth가 아닌 평가**가 포함됨 (pseudo-label / LLM Judge)
- CounselChat은 30샘플로 **표본이 작음**
- Conditional 구조는 “연산 완전 절약”이 아니라, **Revision/Recheck만 조건부** (Gatekeeper까지는 수행)
- JSON 출력은 사용자 UX가 아니라 **파이프라인 내부 통신/평가/로깅**을 위한 구조화 포맷

