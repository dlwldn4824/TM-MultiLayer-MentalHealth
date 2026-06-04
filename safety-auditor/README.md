# Mental Health LLM Safety Auditor

Single LLM vs Agent Structure 비교·Judge 평가·리포트 생성 웹 앱 (Ollama 로컬 모델).

## Stack

- **Frontend:** Next.js 15, TypeScript, TailwindCSS, Recharts
- **Backend:** FastAPI, SQLite
- **LLM/Judge:** Ollama 로컬 4모델 (OpenAI 미사용)

## Quick Start

### 1. Backend

```bash
cd safety-auditor/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd safety-auditor/frontend
npm install
cp .env.local.example .env.local
npm run dev
```

브라우저: http://localhost:3000

- **Home:** `/`
- **Compare:** `/compare`

## Features

| 기능 | 설명 |
|------|------|
| Compare | Single vs Agent 응답 + runtime/tokens |
| Structure | Emotion→Strategy→Response→Safety, Router, Bidirectional 등 |
| Judge | Safety, Empathy, Trust, Helpfulness, Faithfulness, Coherence (1–5) |
| Charts | Table, Bar, Radar |
| Pipeline View | 단계별 runtime / prompt / output |
| Insights | 자동 요약 문장 |
| Export | Markdown, JSON (PDF placeholder) |
| CSV Upload | D_OFF, A_OFF, Router, Bidirectional 실험 CSV 비교 차트 |

## API

- `POST /api/compare`
- `POST /api/insights`
- `POST /api/report`
- `POST /api/upload-csv`
- `GET /api/csv-comparison`
- `GET /api/evaluations`

DB: `backend/data/auditor.db`

## Ollama 모델 (실험과 동일)

| 모델 | 용도 |
|------|------|
| `qwen2.5:7b` | 기본 generation (Single + Agent) |
| `llama3.1:8b` | 기본 Judge |
| `gemma2:9b` | generation 선택 |
| `mistral:7b` | generation 선택 |

환경 변수:

```bash
export AUDITOR_LLM_MODE=ollama   # ollama | mock | auto
export OLLAMA_BASE_URL=http://localhost:11434
export AUDITOR_DEFAULT_MODEL=qwen2.5:7b
export AUDITOR_JUDGE_MODEL=llama3.1:8b
export OLLAMA_TIMEOUT=600
```

`AUDITOR_LLM_MODE=mock` 이면 API 없이 목업으로 동작.
