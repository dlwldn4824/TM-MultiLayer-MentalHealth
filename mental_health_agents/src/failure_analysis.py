"""Phase A: Build failure dataset from experiment outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config
from src.database import Database
from src.training_io import write_jsonl
from src.training_paths import training_paths
from src.utils import extract_json

logger = logging.getLogger(__name__)

FAILURE_TYPES = (
    "hallucination",
    "unsafe_response",
    "risk_mismatch",
    "missed_high_risk",
    "json_parse_failure",
    "unsupported_diagnosis",
)


def _parse_safety_flags(output_json: Any) -> dict[str, bool]:
    if isinstance(output_json, dict):
        flags = output_json.get("safety_flags", {})
        if isinstance(flags, dict):
            return {k: bool(v) for k, v in flags.items()}
    return {}


def _collect_json_failures(db: Database) -> set[tuple[str, str]]:
    failed: set[tuple[str, str]] = set()
    for row in db.fetch_agent_outputs():
        parsed = row.get("output_json")
        raw = row.get("raw_output") or ""
        if parsed is None and raw and extract_json(raw) is None:
            failed.add((row["case_id"], row["experiment_name"]))
        if not parsed and not raw.strip():
            failed.add((row["case_id"], row["experiment_name"]))
    return failed


def detect_failures(
    db: Database | None = None,
    predictions_csv: str | None = None,
) -> list[dict[str, Any]]:
    cfg = load_config()
    db = db or Database()
    csv_path = predictions_csv or cfg["output"]["predictions_csv"]

    cases_by_id = {c["case_id"]: c for c in db.fetch_cases()}
    json_failures = _collect_json_failures(db)

    preds: list[dict[str, Any]]
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        preds = df.to_dict(orient="records")
        for p in preds:
            p["hallucination_flag"] = str(p.get("hallucination_flag", False)).lower() in (
                "true",
                "1",
                "yes",
            )
            p["unsafe_flag"] = str(p.get("unsafe_flag", False)).lower() in ("true", "1", "yes")
    else:
        preds = db.fetch_predictions()

    failures: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_failure(
        case_id: str,
        experiment_name: str,
        failure_type: str,
        pred_row: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        key = (case_id, experiment_name, failure_type)
        if key in seen:
            return
        seen.add(key)
        case = cases_by_id.get(case_id, {})
        text = case.get("text", "")
        gt_risk = pred_row.get("true_risk") or case.get("risk_label", "")
        pred_risk = pred_row.get("predicted_risk", "")
        trace = db.fetch_agent_trace(case_id, experiment_name)
        evidence = []
        raw_parts = []
        for t in trace:
            ev = t.get("evidence") or []
            if isinstance(ev, list):
                evidence.extend(ev)
            raw_parts.append(t.get("raw_output") or "")

        agent_outputs = db.fetch_agent_outputs(case_id=case_id, experiment_name=experiment_name)
        unsupported = False
        for ao in agent_outputs:
            flags = _parse_safety_flags(ao.get("output_json"))
            if flags.get("unsupported_diagnosis"):
                unsupported = True

        row = {
            "id": f"{case_id}__{experiment_name}__{failure_type}",
            "text": text,
            "experiment_name": experiment_name,
            "model": pred_row.get("model", ""),
            "failure_type": failure_type,
            "ground_truth": gt_risk,
            "prediction": pred_risk,
            "agent_trace": trace,
            "retrieved_evidence": evidence[:5],
            "raw_output": "\n---\n".join(raw_parts)[:4000],
        }
        if extra:
            row.update(extra)
        failures.append(row)

    for p in preds:
        case_id = str(p.get("case_id", ""))
        exp = str(p.get("experiment_name", ""))
        if not case_id or not exp:
            continue

        gt = str(p.get("true_risk", "")).lower()
        pr = str(p.get("predicted_risk", "")).lower()

        if p.get("hallucination_flag"):
            add_failure(case_id, exp, "hallucination", p)
        if p.get("unsafe_flag"):
            add_failure(case_id, exp, "unsafe_response", p)
        if gt and pr and gt != pr:
            add_failure(case_id, exp, "risk_mismatch", p)
        if gt == "high" and pr != "high":
            add_failure(case_id, exp, "missed_high_risk", p)
        if (case_id, exp) in json_failures:
            add_failure(case_id, exp, "json_parse_failure", p)

        for ao in db.fetch_agent_outputs(case_id=case_id, experiment_name=exp):
            flags = _parse_safety_flags(ao.get("output_json"))
            if flags.get("unsupported_diagnosis"):
                add_failure(case_id, exp, "unsupported_diagnosis", p)
                break

    return failures


def build_failure_dataset(output_path: Path | None = None) -> int:
    paths = training_paths()
    out = output_path or paths["failure"] / "failure_cases.jsonl"
    failures = detect_failures()
    n = write_jsonl(out, failures)
    logger.info("Wrote %d failure cases to %s", n, out)
    return n
