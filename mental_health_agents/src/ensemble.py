"""Weighted voting ensemble over multiple model risk predictions."""

from __future__ import annotations

from collections import Counter
from typing import Any

RISK_SCORES = {"low": 0.0, "medium": 1.0, "high": 2.0}
SCORE_TO_RISK = [(0.6, "low"), (1.4, "medium"), (2.0, "high")]


def risk_to_score(risk: str) -> float:
    return RISK_SCORES.get(str(risk).lower(), 0.0)


def score_to_risk(score: float) -> str:
    if score <= 0.6:
        return "low"
    if score <= 1.4:
        return "medium"
    return "high"


def majority_vote(risks: list[str]) -> str:
    if not risks:
        return "low"
    counts = Counter(risks)
    return counts.most_common(1)[0][0]


def weighted_risk_vote(
    model_risks: dict[str, str],
    weights: dict[str, float],
) -> str:
    total_w = sum(weights.values()) or 1.0
    score = sum(risk_to_score(model_risks[m]) * weights.get(m, 0.0) for m in model_risks) / total_w
    return score_to_risk(score)


def performance_weights(
    models: list[str],
    model_metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    raw = {}
    for m in models:
        met = model_metrics.get(m, {})
        raw[m] = max(
            0.01,
            met.get("high_risk_recall", 0) + met.get("symptom_f1", 0) - met.get("unsafe_flag_rate", 0),
        )
    s = sum(raw.values()) or 1.0
    return {m: raw[m] / s for m in models}


def safety_weights(
    models: list[str],
    model_metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    raw = {}
    for m in models:
        met = model_metrics.get(m, {})
        raw[m] = max(0.01, 1.0 - met.get("unsafe_flag_rate", 0.5) + met.get("high_risk_recall", 0))
    s = sum(raw.values()) or 1.0
    return {m: raw[m] / s for m in models}


def equal_weights(models: list[str]) -> dict[str, float]:
    n = len(models) or 1
    return {m: 1.0 / n for m in models}


def combine_case_predictions(
    per_model_outputs: dict[str, dict[str, Any]],
    method: str,
    model_metrics: dict[str, dict[str, float]] | None = None,
    custom_weights: list[float] | None = None,
    model_order: list[str] | None = None,
) -> dict[str, Any]:
    """Combine per-model final outputs for one case."""
    models = model_order or list(per_model_outputs.keys())
    risks = {m: per_model_outputs[m].get("risk_level", "low") for m in models if m in per_model_outputs}

    if method == "majority_vote":
        final_risk = majority_vote(list(risks.values()))
    elif method == "equal_weight":
        w = equal_weights(list(risks.keys()))
        final_risk = weighted_risk_vote(risks, w)
    elif method == "performance_weight":
        w = performance_weights(list(risks.keys()), model_metrics or {})
        final_risk = weighted_risk_vote(risks, w)
    elif method == "safety_weight":
        w = safety_weights(list(risks.keys()), model_metrics or {})
        final_risk = weighted_risk_vote(risks, w)
    elif method == "custom_weight" and custom_weights and model_order:
        w = {m: custom_weights[i] for i, m in enumerate(model_order) if i < len(custom_weights)}
        final_risk = weighted_risk_vote(risks, w)
    else:
        final_risk = majority_vote(list(risks.values()))

    all_symptoms: set[str] = set()
    for m in models:
        out = per_model_outputs.get(m, {})
        for s in out.get("symptoms", []) or []:
            all_symptoms.add(s)

    responses = [per_model_outputs[m].get("response", "") for m in models if per_model_outputs.get(m)]
    return {
        "symptoms": sorted(all_symptoms),
        "risk_level": final_risk,
        "evidence": [],
        "response": responses[0] if responses else "",
        "uncertainty": f"ensemble:{method}",
        "safety_flags": {
            "unsupported_diagnosis": False,
            "fabricated_symptom": False,
            "unsafe_advice": False,
            "missed_self_harm_risk": False,
        },
    }
