"""Shared utilities."""

from __future__ import annotations

import json
import re
from typing import Any


def ensure_dir(path: str) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)


def extract_json(text: str) -> dict[str, Any] | None:
    """Try to parse JSON from LLM output (raw or fenced)."""
    text = text.strip()
    if not text:
        return None

    # Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Markdown code block
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # First {...} block
    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        try:
            obj = json.loads(brace.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def normalize_risk(level: str | None) -> str:
    if not level:
        return "low"
    level = str(level).strip().lower()
    if level in ("low", "medium", "high"):
        return level
    if "high" in level:
        return "high"
    if "med" in level:
        return "medium"
    return "low"


def normalize_symptoms(symptoms: Any) -> list[str]:
    if symptoms is None:
        return []
    if isinstance(symptoms, str):
        symptoms = [symptoms]
    if not isinstance(symptoms, list):
        return []
    return sorted({str(s).strip().lower().replace(" ", "_") for s in symptoms if s})


def default_output_schema() -> dict[str, Any]:
    return {
        "symptoms": [],
        "risk_level": "low",
        "evidence": [],
        "response": "",
        "uncertainty": "",
        "safety_flags": {
            "unsupported_diagnosis": False,
            "fabricated_symptom": False,
            "unsafe_advice": False,
            "missed_self_harm_risk": False,
        },
    }


def merge_output(parsed: dict[str, Any] | None, raw: str = "") -> dict[str, Any]:
    out = default_output_schema()
    if not parsed:
        out["response"] = raw[:2000] if raw else "Failed to parse model output."
        out["uncertainty"] = "high — JSON parse failed"
        return out

    out["symptoms"] = normalize_symptoms(parsed.get("symptoms"))
    out["risk_level"] = normalize_risk(parsed.get("risk_level"))
    ev = parsed.get("evidence", [])
    out["evidence"] = ev if isinstance(ev, list) else ([ev] if ev else [])
    out["response"] = str(parsed.get("response", parsed.get("final_response", "")))
    out["uncertainty"] = str(parsed.get("uncertainty", ""))
    flags = parsed.get("safety_flags", {})
    if isinstance(flags, dict):
        for k in out["safety_flags"]:
            if k in flags:
                out["safety_flags"][k] = bool(flags[k])
    return out
