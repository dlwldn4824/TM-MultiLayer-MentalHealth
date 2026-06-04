"""Mental Health LLM Safety Auditor — FastAPI backend."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.database import get_connection, init_db, row_to_dict
from app.schemas import (
    CompareRequest,
    CompareResponse,
    InsightRequest,
    InsightResponse,
    ReportRequest,
)
from app.services.csv_upload import merge_chart_data, parse_csv_content
from app.config import EXPERIMENT_MODELS, LLM_MODE
from app.services.insights import generate_insights
from app.services.llm_router import generate_agent, generate_plain, use_ollama
from app.services.mock_llm import STRUCTURE_PIPELINES
from app.services.ollama_client import list_ollama_models
from app.services.ollama_judge import judge_response
from app.services.report import generate_json_report, generate_markdown

app = FastAPI(
    title="Mental Health LLM Safety Auditor",
    version="0.1.0",
    description="Compare Single LLM vs Agent Structure with judge evaluation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict[str, Any]:
    installed = list_ollama_models()
    return {
        "status": "ok",
        "llm_mode": LLM_MODE,
        "use_ollama": use_ollama(),
        "ollama_models_installed": installed,
        "experiment_models": [m["id"] for m in EXPERIMENT_MODELS],
    }


@app.get("/api/structures")
def list_structures() -> dict[str, Any]:
    return {
        "structures": [
            {
                "id": k,
                "label": " → ".join(v),
                "steps": v,
            }
            for k, v in STRUCTURE_PIPELINES.items()
        ]
    }


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    installed = set(list_ollama_models())
    generation = [m["id"] for m in EXPERIMENT_MODELS if m["role"] == "generation"]
    judges = [m["id"] for m in EXPERIMENT_MODELS if m["role"] == "judge"]
    all_ids = [m["id"] for m in EXPERIMENT_MODELS]
    return {
        "models": generation if generation else all_ids,
        "judges": list(dict.fromkeys(judges + all_ids)),
        "model_details": EXPERIMENT_MODELS,
        "ollama_available": {m: m in installed for m in [x["id"] for x in EXPERIMENT_MODELS]},
        "use_ollama": use_ollama(),
        "llm_mode": LLM_MODE,
    }


@app.post("/api/compare", response_model=CompareResponse)
async def compare(req: CompareRequest) -> CompareResponse:
    if not req.query.strip():
        raise HTTPException(400, "query is required")

    plain = await generate_plain(req.query, req.model)
    agent, steps = await generate_agent(req.query, req.model, req.structure)

    plain_scores = await judge_response(
        req.query, plain.text, provider=req.judge_provider
    )
    agent_scores = await judge_response(
        req.query, agent.text, provider=req.judge_provider
    )

    eval_id: int | None = None
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO evaluations (
                query, model, structure, plain_response, agent_response,
                plain_scores, agent_scores, runtime_plain, runtime_agent, pipeline_trace
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                req.query,
                req.model,
                req.structure,
                plain.text,
                agent.text,
                plain_scores.model_dump_json(),
                agent_scores.model_dump_json(),
                plain.runtime_seconds,
                agent.runtime_seconds,
                json.dumps([s.model_dump() for s in steps], ensure_ascii=False),
            ),
        )
        conn.commit()
        eval_id = int(cur.lastrowid)

    return CompareResponse(
        query=req.query,
        model=req.model,
        structure=req.structure,
        plain=plain,
        agent=agent,
        pipeline_steps=steps,
        plain_scores=plain_scores,
        agent_scores=agent_scores,
        evaluation_id=eval_id,
    )


@app.post("/api/insights", response_model=InsightResponse)
def insights(req: InsightRequest) -> InsightResponse:
    return generate_insights(req)


@app.post("/api/report")
def report(req: ReportRequest) -> dict[str, Any]:
    if req.format == "json":
        return {"format": "json", "content": generate_json_report(req)}
    return {"format": "markdown", "content": generate_markdown(req)}


@app.get("/api/evaluations")
def list_evaluations(limit: int = 20) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM evaluations ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row_to_dict(r) for r in rows if r]


@app.get("/api/evaluations/{eval_id}")
def get_evaluation(eval_id: int) -> dict[str, Any]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM evaluations WHERE id = ?", (eval_id,)
        ).fetchone()
    out = row_to_dict(row)
    if not out:
        raise HTTPException(404, "evaluation not found")
    return out


@app.post("/api/upload-csv")
async def upload_csv(
    label: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    content = await file.read()
    try:
        row_count, summary, chart_data = parse_csv_content(content, label)
    except Exception as e:
        raise HTTPException(400, f"Invalid CSV: {e}") from e

    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO csv_uploads (label, filename, row_count, summary_json)
            VALUES (?, ?, ?, ?)
            """,
            (label, file.filename or "upload.csv", row_count, json.dumps(summary)),
        )
        conn.commit()
        upload_id = int(cur.lastrowid)

    all_uploads = _load_all_csv_uploads()

    return {
        "upload_id": upload_id,
        "label": label,
        "row_count": row_count,
        "summary": summary,
        "chart_data": chart_data,
        "comparison_chart": merge_chart_data(all_uploads),
    }


def _load_all_csv_uploads() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT label, summary_json FROM csv_uploads").fetchall()
    uploads = []
    for r in rows:
        summ = json.loads(r["summary_json"] or "{}")
        cd = []
        for col in ("safety", "empathy", "trust", "helpfulness", "faithfulness"):
            if f"mean_{col}" in summ:
                cd.append({"metric": col, "value": summ[f"mean_{col}"], "label": r["label"]})
        uploads.append({"label": r["label"], "chart_data": cd})
    return uploads


@app.get("/api/csv-comparison")
def csv_comparison() -> dict[str, Any]:
    uploads = _load_all_csv_uploads()
    return {
        "uploads": uploads,
        "comparison_chart": merge_chart_data(uploads),
    }
