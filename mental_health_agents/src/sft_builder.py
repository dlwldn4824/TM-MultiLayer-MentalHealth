"""Phase B/C: Agent output → SFT datasets (flat, chain, alpaca, chatml, task splits)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config import load_config
from src.database import Database
from src.prompts import RESEARCH_SYSTEM
from src.training_io import write_jsonl, write_splits
from src.training_paths import training_paths

logger = logging.getLogger(__name__)

FLAT_INSTRUCTION = "Perform psychiatric reasoning for research evaluation. Return structured JSON."
CHAIN_INSTRUCTION = "Perform stepwise psychiatric reasoning for research evaluation."
TASK_INSTRUCTIONS = {
    "symptom_extraction": "Extract mental health symptoms from the patient text as JSON.",
    "risk_classification": "Assess suicide/self-harm risk level (low/medium/high) with evidence.",
    "safety_reasoning": "Verify safety issues in the psychiatric reasoning draft.",
    "full_reasoning": FLAT_INSTRUCTION,
}

AGENT_STEP_MAP = {
    "symptom_extraction": "symptom_extraction",
    "symptom_extraction_skipped": "symptom_extraction",
    "risk_assessment": "risk_assessment",
    "risk_assessment_skipped": "risk_assessment",
    "safety_verification": "safety_verification",
    "safety_verification_skipped": "safety_verification",
    "consensus": "consensus",
    "consensus_skipped": "consensus",
}


def _final_output_from_trace(trace: list[dict[str, Any]], pred: dict[str, Any]) -> dict[str, Any]:
    for t in reversed(trace):
        if t.get("agent") == "consensus" and isinstance(t.get("output"), dict):
            return t["output"]
        parsed = t.get("output_json") or t.get("parsed")
        if t.get("agent") == "consensus" and parsed:
            return parsed if isinstance(parsed, dict) else {}
    return {
        "symptoms": pred.get("predicted_symptoms", []),
        "risk_level": pred.get("predicted_risk", "low"),
        "response": pred.get("final_response", ""),
    }


def _chain_from_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    chain: dict[str, Any] = {}
    for t in trace:
        agent = t.get("agent_name") or t.get("agent") or ""
        step = AGENT_STEP_MAP.get(agent)
        if not step:
            continue
        payload = t.get("output_json") or t.get("parsed") or {}
        if step == "symptom_extraction":
            chain[step] = {
                "symptoms": t.get("symptoms") or payload.get("symptoms", []),
                "evidence": t.get("evidence") or payload.get("evidence", []),
            }
        elif step == "risk_assessment":
            chain[step] = {
                "risk_level": t.get("risk_level") or payload.get("risk_level", "low"),
                "evidence": t.get("evidence") or payload.get("evidence", []),
            }
        elif step == "safety_verification":
            chain[step] = {
                "safety_flags": t.get("safety_flags") or payload.get("safety_flags", {}),
            }
        elif step == "consensus":
            out = t.get("output") or payload
            chain[step] = out if isinstance(out, dict) else {}
    return chain


def build_agent_output_rows(db: Database | None = None) -> tuple[list[dict], list[dict]]:
    db = db or Database()
    multi_structures = ("multi_agent", "multi_agent_rag")
    preds = [p for p in db.fetch_predictions() if (p.get("structure") or "") in multi_structures]
    if not preds:
        preds = [
            p
            for p in db.fetch_predictions()
            if "multi_agent" in str(p.get("experiment_name", ""))
        ]

    flat_rows: list[dict] = []
    chain_rows: list[dict] = []

    for p in preds:
        case_id = p["case_id"]
        exp = p["experiment_name"]
        case = db.fetch_case(case_id)
        if not case:
            continue
        text = case["text"]
        trace = db.fetch_agent_trace(case_id, exp)
        if not trace:
            continue

        final_json = _final_output_from_trace(trace, p)
        flat_rows.append(
            {
                "instruction": FLAT_INSTRUCTION,
                "input": text,
                "output": json.dumps(final_json, ensure_ascii=False),
            }
        )
        chain_out = _chain_from_trace(trace)
        if chain_out:
            chain_rows.append(
                {
                    "instruction": CHAIN_INSTRUCTION,
                    "input": text,
                    "output": chain_out,
                }
            )

    return flat_rows, chain_rows


def to_alpaca(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "instruction": r["instruction"],
            "input": r["input"],
            "output": r["output"] if isinstance(r["output"], str) else json.dumps(r["output"], ensure_ascii=False),
        }
        for r in rows
    ]


def to_chatml(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        assistant = r["output"] if isinstance(r["output"], str) else json.dumps(r["output"], ensure_ascii=False)
        out.append(
            {
                "messages": [
                    {"role": "system", "content": RESEARCH_SYSTEM},
                    {"role": "user", "content": f"{r['instruction']}\n\nPatient text:\n{r['input']}"},
                    {"role": "assistant", "content": assistant},
                ]
            }
        )
    return out


def build_task_specific_rows(flat_rows: list[dict]) -> dict[str, list[dict]]:
    by_task: dict[str, list[dict]] = {k: [] for k in TASK_INSTRUCTIONS}
    for r in flat_rows:
        try:
            obj = json.loads(r["output"]) if isinstance(r["output"], str) else r["output"]
        except json.JSONDecodeError:
            continue
        text = r["input"]
        by_task["full_reasoning"].append(
            {"instruction": TASK_INSTRUCTIONS["full_reasoning"], "input": text, "output": json.dumps(obj, ensure_ascii=False)}
        )
        by_task["symptom_extraction"].append(
            {
                "instruction": TASK_INSTRUCTIONS["symptom_extraction"],
                "input": text,
                "output": json.dumps({"symptoms": obj.get("symptoms", [])}, ensure_ascii=False),
            }
        )
        by_task["risk_classification"].append(
            {
                "instruction": TASK_INSTRUCTIONS["risk_classification"],
                "input": text,
                "output": json.dumps(
                    {"risk_level": obj.get("risk_level", "low"), "evidence": obj.get("evidence", [])},
                    ensure_ascii=False,
                ),
            }
        )
        by_task["safety_reasoning"].append(
            {
                "instruction": TASK_INSTRUCTIONS["safety_reasoning"],
                "input": text,
                "output": json.dumps({"safety_flags": obj.get("safety_flags", {})}, ensure_ascii=False),
            }
        )
    return by_task


def build_sft_datasets(smoke_limit: int | None = None) -> dict[str, int]:
    cfg = load_config()
    paths = training_paths()
    sft_dir = paths["sft"]
    sft_dir.mkdir(parents=True, exist_ok=True)

    flat, chain = build_agent_output_rows()
    if smoke_limit:
        flat, chain = flat[:smoke_limit], chain[:smoke_limit]

    counts = {
        "sft_flat": write_jsonl(sft_dir / "sft_flat.jsonl", flat),
        "sft_chain": write_jsonl(sft_dir / "sft_chain.jsonl", chain),
    }

    alpaca_flat = to_alpaca(flat)
    alpaca_chain = to_alpaca(chain)
    write_jsonl(sft_dir / "alpaca_flat.jsonl", alpaca_flat)
    write_jsonl(sft_dir / "alpaca_chain.jsonl", alpaca_chain)
    write_jsonl(sft_dir / "chatml_flat.jsonl", to_chatml(flat))
    write_jsonl(sft_dir / "chatml_chain.jsonl", to_chatml(chain))

    ratios = tuple(cfg.get("training", {}).get("split_ratios", [0.8, 0.1, 0.1]))
    seed = cfg.get("training", {}).get("split_seed", 42)
    split_counts = write_splits(sft_dir, "alpaca_flat", alpaca_flat, tuple(ratios), seed)
    counts.update({f"alpaca_flat_{k}": v for k, v in split_counts.items()})

    tasks = build_task_specific_rows(flat)
    for task_name, rows in tasks.items():
        if smoke_limit:
            rows = rows[:smoke_limit]
        write_jsonl(sft_dir / f"task_{task_name}.jsonl", rows)
        write_splits(sft_dir, f"task_{task_name}", rows, tuple(ratios), seed)
        counts[f"task_{task_name}"] = len(rows)

    logger.info("SFT datasets: %s", counts)
    return counts
