"""Rank experiment runs and select top candidates for next phase."""

from __future__ import annotations

from typing import Any


def score_metrics(metrics: dict[str, float], cfg: dict[str, Any] | None = None) -> tuple[float, float, float]:
    """Higher is better. Primary: high_risk_recall, secondary: -unsafe, tie: symptom_f1."""
    return (
        float(metrics.get("high_risk_recall", 0.0)),
        -float(metrics.get("unsafe_flag_rate", 1.0)),
        float(metrics.get("symptom_f1", 0.0)),
    )


def rank_runs(
    run_metrics: list[dict[str, Any]],
    top_k: int = 2,
) -> list[dict[str, Any]]:
    """run_metrics items: {run_config, metrics, experiment_name, ...}"""
    sorted_runs = sorted(
        run_metrics,
        key=lambda r: score_metrics(r.get("metrics", {})),
        reverse=True,
    )
    return sorted_runs[:top_k]


def pick_best(run_metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked = rank_runs(run_metrics, top_k=1)
    return ranked[0] if ranked else None


def extract_structure_winner(best: dict[str, Any]) -> str:
    rc = best.get("run_config", {})
    if isinstance(rc, dict):
        return rc.get("structure", "multi_agent_rag")
    return getattr(rc, "structure", "multi_agent_rag")


def extract_model_winner(best: dict[str, Any]) -> str:
    rc = best.get("run_config", {})
    if isinstance(rc, dict):
        return rc.get("model", "qwen2.5:7b")
    return getattr(rc, "model", "qwen2.5:7b")
