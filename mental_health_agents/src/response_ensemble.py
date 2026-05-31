"""Response-level ensemble strategies for Phase 3 (CounselChat benchmark)."""

from __future__ import annotations

from collections import Counter
from typing import Any

JUDGE_CRITERIA = ("faithfulness", "answer_relevancy", "empathy", "safety")

DEFAULT_JUDGE_WEIGHTS = {
    "faithfulness": 0.35,
    "answer_relevancy": 0.25,
    "empathy": 0.20,
    "safety": 0.20,
}

SAFETY_PRIORITIZED_WEIGHTS = {
    "faithfulness": 0.15,
    "answer_relevancy": 0.15,
    "empathy": 0.15,
    "safety": 0.55,
}

EQUAL_WEIGHTS = {k: 0.25 for k in JUDGE_CRITERIA}

ENSEMBLE_MERGE_PROMPT = """You are synthesizing a single supportive counseling-style reply from multiple candidate responses.
Treat every candidate with equal importance. Preserve accurate, safe, empathetic content.
Do not diagnose or prescribe medication. Return JSON only:
{{"response": "<merged reply>"}}

Client message:
{question}

Candidate responses:
{candidates_block}
"""


def _weighted_score(judge_row: dict[str, float], weights: dict[str, float]) -> float:
    return sum(float(judge_row.get(k, 0.0)) * weights.get(k, 0.0) for k in JUDGE_CRITERIA)


def _total_score(judge_row: dict[str, float]) -> float:
    return sum(float(judge_row.get(k, 0.0)) for k in JUDGE_CRITERIA)


def majority_selection(
    candidates: dict[str, str],
    judge_by_model: dict[str, dict[str, float]],
) -> tuple[str, str, dict[str, Any]]:
    """Pick response that wins the most judge dimensions; tie-break by total score."""
    if not candidates:
        return "", "none", {}
    if len(candidates) == 1:
        model = next(iter(candidates))
        return candidates[model], model, {"votes": {model: len(JUDGE_CRITERIA)}}

    votes: Counter[str] = Counter()
    for crit in JUDGE_CRITERIA:
        winner = max(
            candidates,
            key=lambda m: (judge_by_model.get(m, {}).get(crit, 0.0), _total_score(judge_by_model.get(m, {}))),
        )
        votes[winner] += 1

    best = max(
        candidates,
        key=lambda m: (votes[m], _total_score(judge_by_model.get(m, {}))),
    )
    return candidates[best], best, {"votes": dict(votes)}


def score_weighted_selection(
    candidates: dict[str, str],
    judge_by_model: dict[str, dict[str, float]],
    weights: dict[str, float],
    model_weights: dict[str, float] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Pick response with highest weighted judge score (optionally scaled by model weights)."""
    if not candidates:
        return "", "none", {}
    if len(candidates) == 1:
        model = next(iter(candidates))
        return candidates[model], model, {"scores": {model: _weighted_score(judge_by_model.get(model, {}), weights)}}

    scores: dict[str, float] = {}
    for model in candidates:
        row = judge_by_model.get(model, {})
        base = _weighted_score(row, weights)
        if model_weights:
            base *= model_weights.get(model, 1.0)
        scores[model] = base

    best = max(candidates, key=lambda m: scores[m])
    return candidates[best], best, {"scores": scores}


def equal_weight_selection(
    candidates: dict[str, str],
    judge_by_model: dict[str, dict[str, float]],
) -> tuple[str, str, dict[str, Any]]:
    return score_weighted_selection(candidates, judge_by_model, EQUAL_WEIGHTS)


def judge_weighted_selection(
    candidates: dict[str, str],
    judge_by_model: dict[str, dict[str, float]],
    model_weights: dict[str, float] | None = None,
    criterion_weights: dict[str, float] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    weights = criterion_weights or DEFAULT_JUDGE_WEIGHTS
    return score_weighted_selection(candidates, judge_by_model, weights, model_weights=model_weights)


def safety_prioritized_selection(
    candidates: dict[str, str],
    judge_by_model: dict[str, dict[str, float]],
    model_weights: dict[str, float] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    return score_weighted_selection(
        candidates,
        judge_by_model,
        SAFETY_PRIORITIZED_WEIGHTS,
        model_weights=model_weights,
    )


def apply_ensemble(
    method: str,
    candidates: dict[str, str],
    judge_by_model: dict[str, dict[str, float]],
    model_weights: dict[str, float] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    if method == "majority_selection":
        return majority_selection(candidates, judge_by_model)
    if method == "equal_weight":
        return equal_weight_selection(candidates, judge_by_model)
    if method == "judge_weighted":
        return judge_weighted_selection(candidates, judge_by_model, model_weights=model_weights)
    if method == "safety_prioritized":
        return safety_prioritized_selection(candidates, judge_by_model, model_weights=model_weights)
    raise ValueError(f"Unknown ensemble method: {method}")


def format_candidates_block(candidates: dict[str, str]) -> str:
    lines = []
    for i, (model, text) in enumerate(candidates.items(), 1):
        lines.append(f"--- Candidate {i} ({model}) ---\n{text[:2500]}")
    return "\n\n".join(lines)


def model_performance_weights(
    model_metrics: dict[str, dict[str, float]],
    models: list[str],
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for model in models:
        m = model_metrics.get(model, {})
        raw[model] = max(
            0.01,
            m.get("judge_faithfulness", 0.0)
            + m.get("judge_safety", 0.0)
            + m.get("judge_empathy", 0.0),
        )
    total = sum(raw.values()) or 1.0
    return {model: raw[model] / total for model in models}
