"""Phase E: DPO preference pairs from experiment outcomes."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.config import load_config
from src.database import Database
from src.training_io import write_jsonl, write_splits
from src.training_paths import training_paths

logger = logging.getLogger(__name__)


def _score_response(pred: dict[str, Any], case: dict[str, Any]) -> float:
    """Higher = better (chosen candidate)."""
    gt = str(case.get("risk_label", pred.get("true_risk", ""))).lower()
    pr = str(pred.get("predicted_risk", "")).lower()
    score = 0.0
    if not pred.get("unsafe_flag"):
        score += 2.0
    if not pred.get("hallucination_flag"):
        score += 2.0
    if gt and pr == gt:
        score += 3.0
    if gt == "high" and pr == "high":
        score += 2.0
    resp = pred.get("final_response") or ""
    if resp and len(resp) > 30:
        score += 0.5
    if "diagnose" in resp.lower() and "may" not in resp.lower():
        score -= 1.0
    return score


def _format_assistant(pred: dict[str, Any]) -> str:
    obj = {
        "symptoms": pred.get("predicted_symptoms", []),
        "risk_level": pred.get("predicted_risk", "low"),
        "response": pred.get("final_response", ""),
    }
    return json.dumps(obj, ensure_ascii=False)


def build_dpo_pairs(db: Database | None = None) -> list[dict[str, Any]]:
    db = db or Database()
    cases = {c["case_id"]: c for c in db.fetch_cases()}
    by_case: dict[str, list[dict]] = {}
    for p in db.fetch_predictions():
        by_case.setdefault(p["case_id"], []).append(p)

    pairs: list[dict[str, Any]] = []
    for case_id, preds in by_case.items():
        case = cases.get(case_id)
        if not case or len(preds) < 2:
            continue
        text = case["text"]
        ranked = sorted(preds, key=lambda p: _score_response(p, case), reverse=True)
        chosen = ranked[0]
        rejected = ranked[-1]
        if _score_response(chosen, case) <= _score_response(rejected, case):
            continue
        prompt = (
            "Perform psychiatric reasoning for research. Return JSON with symptoms, "
            f"risk_level, and response.\n\nPatient text:\n{text}"
        )
        pairs.append(
            {
                "prompt": prompt,
                "chosen": _format_assistant(chosen),
                "rejected": _format_assistant(rejected),
                "case_id": case_id,
                "chosen_experiment": chosen.get("experiment_name"),
                "rejected_experiment": rejected.get("experiment_name"),
            }
        )
    return pairs


def build_dpo_dataset(smoke_limit: int | None = None) -> int:
    cfg = load_config()
    paths = training_paths()
    pairs = build_dpo_pairs()
    if smoke_limit:
        pairs = pairs[:smoke_limit]
    out = paths["dpo"] / "dpo_pairs.jsonl"
    paths["dpo"].mkdir(parents=True, exist_ok=True)
    n = write_jsonl(out, pairs)
    ratios = tuple(cfg.get("training", {}).get("split_ratios", [0.8, 0.1, 0.1]))
    seed = cfg.get("training", {}).get("split_seed", 42)
    write_splits(paths["dpo"], "dpo_pairs", pairs, ratios, seed)
    logger.info("Wrote %d DPO pairs to %s", n, out)
    return n
