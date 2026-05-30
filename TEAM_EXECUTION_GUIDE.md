# 팀 역할별 실행 가이드

---

# 2번 — RAG / Knowledge 담당

## 역할

공식 정신건강 지식베이스 구축 및 Retrieval 품질 개선 담당.

주요 업무:

* Official KB 구축
* ChromaDB 인덱싱
* Retrieval 품질 평가
* Retrieval parameter tuning

---

## 1. 공식 KB 다운로드 / 구축

### 공식 문서 fetch

실행:

```bash
python scripts/fetch_official_knowledge.py
```

확인사항:

* NIMH 문서 다운로드 성공 여부
* WHO 문서 다운로드 성공 여부
* NICE guideline 다운로드 성공 여부

---

### Knowledge DB 생성

실행:

```bash
python scripts/build_knowledge_db.py
```

확인사항:

* chunk 생성 개수
* embedding 생성 성공 여부
* ChromaDB 저장 성공 여부
* metadata 정상 저장 여부

---

## 2. Retrieval 검증

### Retrieval Smoke Test

실행:

```bash
python scripts/test_retrieval.py
```

확인할 항목:

```text
retrieval success
top-k 검색 정상 여부
official KB lookup 정상 여부
empty retrieval 여부
```

---

### Retrieval Parameter 실험

비교 설정:

```text
top_k = 1 / 3 / 5

chunk_size = 300 / 500 / 800

overlap = 50 / 100
```

실행 예시:

```bash
python scripts/evaluate_retrieval.py --top-k 3 --chunk-size 500
```

```bash
python scripts/evaluate_retrieval.py --top-k 5 --chunk-size 500
```

```bash
python scripts/evaluate_retrieval.py --top-k 3 --chunk-size 300
```

---

## 산출물

저장 위치:

```text
outputs/rag/

retrieval_scores.csv
kb_summary.md
retrieval_analysis.md
```

---

---

# 3번 — Evaluation / Judge 담당

## 역할

평가 metric 및 LLM Judge pipeline 담당.

주요 업무:

* BERTScore / ROUGE 검증
* Judge prompt 관리
* Judge execution 검증
* Runtime metric 측정
* SQLite logging 확인

---

## 1. Reference Evaluation 확인

실행:

```bash
python scripts/evaluate_reference_metrics.py
```

검증 항목:

```text
BERTScore 정상 계산 여부
ROUGE-L 정상 계산 여부
reference answer 매칭 여부
```

---

## 2. LLM Judge 실행

Judge metric:

```text
Faithfulness
Answer Relevancy
Empathy
Safety
```

실행:

```bash
python scripts/run_judge_eval.py
```

검증 항목:

```text
Judge prompt 정상 실행 여부
0~5 score 출력 여부
JSON parse 성공 여부
judge failure 존재 여부
```

---

## 3. Runtime Metric 측정

실행:

```bash
python scripts/runtime_eval.py
```

확인사항:

```text
Average Runtime
JSON Success Rate
Inference Stability
```

---

## 4. SQLite Logging 검증

실행:

```bash
python scripts/check_sqlite_logging.py
```

확인사항:

```text
judge_scores 저장 여부
runtime 저장 여부
phase 저장 여부
metrics 저장 여부
```

---

## 산출물

저장 위치:

```text
outputs/eval/

judge_scores.csv
runtime_summary.csv
metric_summary.md
judge_rubric.md
```

---

---

# 4번 — Phase3~6 후속 실험 담당

## 역할

후속 실험 구조 구축 및 실행 담당.

주요 업무:

* Phase3 Ensemble
* Phase4 Weight Search
* Phase5 Ablation
* Phase6 Scaling

---

# Phase3 — Ensemble

## 구현 대상

```text
majority selection
equal weight
judge-weighted
safety-prioritized
```

---

### 실행

Smoke 실행:

```bash
python scripts/run_phase3.py --sample-size 10
```

Full 실행:

```bash
python scripts/run_phase3.py
```

---

### 산출물

```text
outputs/phase3/

ensemble_results.csv
ensemble_summary.md
```

---

# Phase4 — Weight Search

## 비교 configuration

```text
Faithfulness-heavy

Safety-heavy

Balanced

Latency-aware
```

---

### 실행

```bash
python scripts/run_phase4.py
```

---

### 산출물

```text
outputs/phase4/

weight_search_results.csv
weight_summary.md
optimized_config.yaml
```

---

# Phase5 — Ablation

## 비교 대상

```text
Full Pipeline

no_retrieval

no_safety_agent

no_reasoning_split

optional_consensus
```

---

### 실행

Smoke:

```bash
python scripts/run_phase5.py --sample-size 10
```

Full:

```bash
python scripts/run_phase5.py
```

---

### 산출물

```text
outputs/phase5/

ablation_metrics.csv
ablation_summary.md
pipeline_trace_logs/
```

---

# Phase6 — Scaling

## Benchmark 규모

```text
30
100
300
1000
```

---

### 실행

30 sample:

```bash
python scripts/run_phase6.py --sample-size 30
```

100 sample:

```bash
python scripts/run_phase6.py --sample-size 100
```

300 sample:

```bash
python scripts/run_phase6.py --sample-size 300
```

---

### 산출물

```text
outputs/phase6/

scaling_results.csv
scaling_summary.md
```

---

# 공통 환경 세팅 (모든 팀원)

## 환경 활성화

```bash
cd mental_health_agents

source .venv/bin/activate
```

---

## 패키지 설치

```bash
pip install -r requirements.txt
```

---

## Ollama 모델 확인

```bash
ollama list
```

필요시 pull:

```bash
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull gemma2:9b
```

---

## 빠른 정상 동작 확인

```bash
python scripts/run_phase0.py --sample-size 5
```

PASS 확인 후 본인 작업 진행.
