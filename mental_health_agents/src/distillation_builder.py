"""Phase G: Multi-Agent teacher → single student distillation dataset."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.config import load_config
from src.data_loader import load_cases
from src.database import Database
from src.experiments import run_case_with_config
from src.ollama_client import OllamaClient
from src.rag import KnowledgeRAG
from src.run_config import RunConfig
from src.training_io import write_jsonl, write_splits
from src.training_paths import training_paths

logger = logging.getLogger(__name__)


def _teacher_output_from_db(db: Database, case_id: str, experiment_name: str) -> dict[str, Any] | None:
    preds = [p for p in db.fetch_predictions() if p["case_id"] == case_id and p["experiment_name"] == experiment_name]
    if not preds:
        return None
    p = preds[0]
    trace = db.fetch_agent_trace(case_id, experiment_name)
    evidence = []
    for t in trace:
        ev = t.get("evidence") or []
        if isinstance(ev, list):
            evidence.extend([str(x) for x in ev])
    return {
        "symptoms": p.get("predicted_symptoms", []),
        "risk_level": p.get("predicted_risk", "low"),
        "response": p.get("final_response", ""),
        "agent_trace": trace,
        "retrieval_context": evidence[:5],
    }


def build_distillation_rows(
    db: Database | None = None,
    run_teacher_if_missing: bool = False,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    cfg = load_config()
    db = db or Database()
    teacher_structure = cfg.get("training", {}).get("distill_teacher_structure", "multi_agent_rag")
    model = cfg["ollama"]["model"]

    teacher_exps = [
        p["experiment_name"]
        for p in db.fetch_predictions()
        if (p.get("structure") or "") == teacher_structure
    ]
    teacher_exp = teacher_exps[0] if teacher_exps else None

    cases = db.fetch_cases()
    if not cases:
        cases = load_cases(sample_size=max_samples or cfg["data"]["sample_size"], force_csv=True)
        for c in cases:
            db.upsert_case(c)

    if max_samples:
        cases = cases[:max_samples]

    rows: list[dict[str, Any]] = []
    client = OllamaClient(cfg)
    rag = KnowledgeRAG(cfg)

    for case in cases:
        case_id = case["case_id"]
        text = case["text"]
        teacher = None
        if teacher_exp:
            teacher = _teacher_output_from_db(db, case_id, teacher_exp)

        if teacher is None and run_teacher_if_missing:
            run_cfg = RunConfig.from_structure(
                phase="distill_teacher",
                structure=teacher_structure,
                model=model,
                sample_size=len(cases),
            )
            result = run_case_with_config(run_cfg, case, client, rag, db)
            teacher = {
                "symptoms": result["output"].get("symptoms", []),
                "risk_level": result["output"].get("risk_level", "low"),
                "response": result["output"].get("response", ""),
                "agent_trace": db.fetch_agent_trace(case_id, run_cfg.experiment_name),
                "retrieval_context": result["output"].get("evidence", []),
            }

        if not teacher:
            continue

        rows.append(
            {
                "input": text,
                "teacher_output": json.dumps(
                    {
                        "symptoms": teacher.get("symptoms", []),
                        "risk_level": teacher.get("risk_level", "low"),
                        "response": teacher.get("response", ""),
                    },
                    ensure_ascii=False,
                ),
                "agent_trace": teacher.get("agent_trace", []),
                "retrieval_context": teacher.get("retrieval_context", []),
            }
        )

    return rows


def build_student_sft_from_distillation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Student learns to imitate teacher single-shot."""
    student_rows = []
    for r in rows:
        student_rows.append(
            {
                "instruction": "Perform psychiatric reasoning (imitate multi-agent teacher).",
                "input": r["input"],
                "output": r["teacher_output"],
            }
        )
    return student_rows


def build_distillation_dataset(
    smoke_limit: int | None = None,
    run_teacher_if_missing: bool = False,
) -> dict[str, int]:
    cfg = load_config()
    paths = training_paths()
    paths["distillation"].mkdir(parents=True, exist_ok=True)

    rows = build_distillation_rows(run_teacher_if_missing=run_teacher_if_missing, max_samples=smoke_limit)
    n = write_jsonl(paths["distillation"] / "teacher_pairs.jsonl", rows)
    student = build_student_sft_from_distillation(rows)
    write_jsonl(paths["distillation"] / "student_sft.jsonl", student)

    ratios = tuple(cfg.get("training", {}).get("split_ratios", [0.8, 0.1, 0.1]))
    seed = cfg.get("training", {}).get("split_seed", 42)
    write_splits(paths["distillation"], "student_sft", student, ratios, seed)

    counts = {"teacher_pairs": n, "student_sft": len(student)}
    logger.info("Distillation datasets: %s", counts)
    return counts
