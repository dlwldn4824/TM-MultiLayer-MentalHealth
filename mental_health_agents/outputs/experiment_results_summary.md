# 실험 결과 종합 및 해석 (2026-05-29)

> **연구용 pseudo-label** 기준 자동 평가입니다. 임상 진단·치료 판단이 아닙니다.

---

## 1. 실험 현황 요약

| # | 실험 | 데이터 | 예측 수 | 상태 |
|---|------|--------|---------|------|
| **A** | 구조 비교 (4조건) — **Smoke** | `sample_cases.csv` × 10건 | 40 | ✅ 완료 |
| **B** | 구조 비교 (4조건) — **본실험** | MentalChat16K × 100건 | 400 | ✅ 완료 (~5.7시간) |
| **C** | Phase 0 파이프라인 검증 | sample × 10, Single LLM | 10 | ✅ 완료 |
| **D** | Training 데이터 구축 (smoke) | inference 로그 | failure 59건 등 | ✅ 완료 (본실험 기준 **재생성 권장**) |
| **E** | Phase 1~6 Sequential | pruning / multi-model | — | ⏳ 미실행 |

- **모델:** `qwen2.5:7b` (Ollama, 로컬)
- **RAG:** 공식 출처 화이트리스트 (NIMH/WHO/NICE) → Chroma **88 chunks**, `official` mode
- **산출물:** `outputs/predictions.csv` (400+행), `outputs/metrics.json`, `outputs/results.sqlite`, `outputs/full_run.log`

---

## 2. 본실험 (MentalChat16K 100건) — 성능 표

| 구조 | Symptom F1 | Symptom Recall | Risk Acc | Risk Macro-F1 | High-Risk Recall | Unsafe Rate ↓ | Hallucination ↓ |
|------|------------|----------------|----------|---------------|------------------|---------------|-----------------|
| Single LLM | 0.19 | 0.39 | 0.08 | 0.05 | 0.00* | **0.15** | 0.00 |
| Single LLM + RAG | 0.37 | 0.67 | 0.14 | 0.09 | 0.00* | 0.71 | 0.00 |
| Multi-Agent | 0.36 | 0.68 | 0.05 | 0.03 | 0.00* | 0.73 | 0.00 |
| Multi-Agent + RAG | **0.41** | **0.74** | **0.18** | **0.11** | 0.00* | 0.93 | 0.00 |

\*본 100건 샘플에 **gold `high` 라벨이 0건**이라 recall 지표는 정의상 0 (아래 §3.1 참고).

*프레임워크 평가 우선순위: High-Risk Recall ↑ → Unsafe ↓ → Symptom F1 ↑*

### 2.1 예측 행동 (본 100건)

| 구조 | pred `low` | pred `medium` | pred `high` |
|------|------------|---------------|-------------|
| Single LLM | 6 | **85** | 9 |
| Single + RAG | 12 | **85** | 3 |
| Multi-Agent | 2 | **98** | 0 |
| Multi + RAG | 16 | **81** | 3 |

- Gold 분포: **low 97 / medium 3 / high 0** (MentalChat 서브셋)
- 공통 패턴: 실제 대부분이 `low`인데 모델은 **`medium`으로 과대 추정** → Risk Accuracy가 전반적으로 낮게 나옴.

---

## 3. 해석 (본실험 중심)

### 3.1 평가 설계상의 한계 (반드시 보고서에 명시)

1. **고위험 recall이 0인 이유**  
   MentalChat 100건에는 pseudo-label `high`가 **한 건도 없음**. `high_risk_recall`은 분모가 없어 모두 0이며, **안전성 벤치마크로는 부적합**한 샘플이다.  
   - 별도 smoke(`case_id=9`, 자살 사고)에서는 Single LLM이 `high`로 맞춤, RAG·Multi는 `medium`으로 **과소평가**한 기록이 있음 (`results.sqlite`).

2. **증상 라벨 희소성**  
   100건 중 **36건은 symptom pseudo-label이 비어 있음**. Recall은 올라가도 Precision이 낮아지기 쉬운 구조 → Symptom F1이 낮게 나오는 것이 자연스러움.

3. **Pseudo-label 한계**  
   MentalChat 대화 + 휴리스틱 라벨러 기준이므로, “모델이 틀렸다”기보다 **기준 자체의 노이즈**가 큼. 특히 risk는 대화 톤·키워드에 민감함.

### 3.2 구조별 의미

| 비교 | 관찰 | 해석 |
|------|------|------|
| **RAG vs No-RAG** | Symptom F1·Recall **상승** (0.19→0.37, Multi 0.36 유지·RAG 0.41) | 공식 지식 검색이 **증상 후보를 더 많이 끌어냄** (recall 우세). |
| **RAG vs No-RAG** | Unsafe rate **급증** (0.15→0.71~0.93) | `safety_flags` 중 하나라도 True면 unsafe. RAG·Multi 파이프라인에서 **보수적 안전 플래그**(`missed_self_harm_risk`, `unsafe_advice` 등)가 자주 켜짐 → “위험하다고 표시”와 “응답 문구가 위험”을 구분해 추가 분석 필요. |
| **Multi vs Single** | 본 100건에서 Symptom F1은 Multi≈Single+RAG, Risk Acc는 Multi+AG **최고(0.18)** | 에이전트 분업이 risk 합의에 다소 유리하나, **medium 쏠림(98%)**으로 차별성이 약함. |
| **Multi + RAG** | Symptom F1·Risk Acc **1위**, Unsafe **최악(0.93)** | **정보 추출·분류 trade-off**: 근거를 붙일수록 “주의 필요” 플래그와 중등도 위험 판정이 동시에 늘어남. |

### 3.3 Smoke(10건) vs 본실험(100건) — 왜 숫자가 크게 다른가?

| 지표 | Smoke (10건, 균형 샘플) | 본실험 (100건, MentalChat) |
|------|-------------------------|----------------------------|
| Symptom F1 최고 | Multi-Agent **0.86** | Multi+RAG **0.41** |
| Risk Acc 최고 | Multi+RAG **0.50** | Multi+RAG **0.18** |
| High-Risk Recall | Single **1.0** (high 1건 포함) | 전부 **0** (high 0건) |
| Unsafe 최저 | Single **0.40** | Single **0.15** |

- Smoke는 **고위험·증상 라벨이 있는 사례를 의도적으로 포함** → Multi-Agent가 유리하게 보임.  
- 본실험은 **저위험 일반 상담 비중이 압도적** → medium 과예측이 accuracy를 깎고, RAG의 recall 상승이 F1로 이어지기 어려움.  
- **소규모 smoke만으로 구조를 고르면 과적합 위험**이 큼. 본 100건 결과가 “실제 배포 분포에 가까운” 쪽에 가깝다고 볼 수 있음.

### 3.4 연구 질문에 대한 답 (프로젝트 관점)

1. **Multi-Agent가 Single보다 나은가?**  
   - Smoke: 증상 F1에서 **명확히 우위**.  
   - 본 100건: 증상은 **Multi+RAG ≈ Single+RAG > Single**, 위험 분류는 **Multi+RAG만 소폭 우위**, 단 absolute 성능은 낮음.

2. **RAG가 도움이 되는가?**  
   - **증상 recall·F1에는 도움** (특히 Single).  
   - **안전 플래그 비율은 악화** → RAG 프롬프트·Safety agent 임계값·플래그 정의 재검토 필요.  
   - 공식 KB 사용으로 unsupported diagnosis hallucination은 0이나, 이는 **키워드 기반 flag**에 한정.

3. **실무적 권장 (현 시점, 추가 튜닝 전)**  
   - **스크리닝 보조(고위험 민감도)**: smoke 기준 Single LLM 참고 + **고위험 전용 hold-out 세트** 필수.  
   - **증상·근거 추출**: Multi-Agent + RAG 후보.  
   - **안전성 우선 배포**: Single LLM(본 100건 unsafe 최저) 또는 Safety agent 보수 규칙 완화 후 재평가.

---

## 4. Smoke 실험 (10건) — 참고 표

| 구조 | Symptom F1 | Risk Acc | High-Risk Recall | Unsafe Rate |
|------|------------|----------|------------------|-------------|
| Single LLM | 0.48 | 0.30 | **1.00** | 0.40 |
| Single LLM + RAG | 0.75 | 0.20 | 0.00 | 0.80 |
| Multi-Agent | **0.86** | 0.30 | 0.00 | 0.70 |
| Multi-Agent + RAG | 0.80 | **0.50** | 0.00 | 1.00 |

※ 당시 RAG 인덱스가 mock/구버전일 수 있음. 공식 인덱스 기준은 **본 100건**이 기준 run.

---

## 5. Phase 0 · Training

### Phase 0
- JSON 파싱 **100%**, `phase0_passed: true`, ~24초/건

### Training (smoke 기준, **본실험 후 재실행 권장**)
| 산출물 | 규모 |
|--------|------|
| `failure_cases.jsonl` | 59건 |
| SFT flat / chain | 각 20건 |
| DPO pairs | 8쌍 |
| LoRA | stub |

본 400건 예측으로 `python src/dataset_builder.py` 재실행 시 failure·DPO 규모가 크게 늘어날 것으로 예상됨 (특히 `risk_mismatch`, `unsafe_response`).

---

## 6. Pruning 제안 (Phase 1 입력)

프레임워크 우선순위(High-Risk Recall → Unsafe → Symptom F1)를 **본 100건에 그대로 적용하기 어렵음** (high 라벨 0). 대안 기준 제안:

| 순위 | 구조 | 근거 |
|------|------|------|
| 1 | **Multi-Agent + RAG** | 본 100건 Symptom F1·Risk Acc 최고 |
| 2 | **Single LLM + RAG** | 증상 recall 우수, unsafe는 Multi 계열보다 낮음 |
| (보류) | Single LLM | 본 100건 분류 지표는 낮으나 smoke·고위험 사례에서 recall 참고 가치 |
| (보류) | Multi-Agent | medium 쏠림·unsafe 높음 — ablation 후 재평가 |

**필수 보완 실험:** 고위험·자살 사고 **균형 hold-out 20~50건**을 고정 벤치로 두고 Phase 1~2 재실행.

---

## 7. 산출 파일

```
outputs/
  predictions.csv              # 400건+ (본실험 + smoke 중복 case_id 주의)
  metrics.json                 # 본 100건 × 4구조 최종 지표
  results.sqlite               # predictions, agent_outputs, cases
  full_run.log                 # 2026-05-29 완료 로그
  experiment_results_summary.md  # 본 문서
  phase_state.json             # Phase 0만 기록

training/                      # smoke 기준 — 재생성 권장
```

---

## 8. 다음 단계

1. `python src/dataset_builder.py` — 본 400건 기준 failure/SFT/DPO 재구축  
2. **고위험 벤치 세트** 추가 (`sample_cases` high/medium + MentalChat 필터) 후 metrics 재계산  
3. `python scripts/run_phases.py --only-phase phase1_structure_comparison`  
4. Safety/RAG 튜닝: `unsafe_advice`·`missed_self_harm_risk` 플래그 조건부 분석 (`agent_outputs` 테이블)  
5. (선택) LoRA mini — `requirements-training.txt` 설치 후 `training_runner.py`

---

## 9. 연구 보고서용 한 줄 요약

> MentalChat16K 100건 본실험에서 **Multi-Agent+RAG가 증상 F1(0.41)과 위험 정확도(0.18)에서 상대적 최고**였으나, 샘플에 고위험 gold가 없어 recall 지표는 판별력이 없었고, **RAG·Multi 계열은 unsafe 플래그 비율이 71~93%로 급증**하는 trade-off가 확인되었다. 소규모 smoke(10건)에서는 Multi-Agent가 F1 0.86으로 우수했으나 **데이터 분포 차이로 순위가 달라져**, 구조 선택은 **균형 hold-out** 없이 결론내리기 어렵다.
