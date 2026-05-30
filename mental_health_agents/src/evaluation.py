"""Compute evaluation metrics and export results."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.config import load_config
from src.database import Database
from src.pruning import pick_best, rank_runs, score_metrics
from src.run_config import RunConfig
from src.utils import ensure_dir, normalize_symptoms

RISK_LABELS = ["low", "medium", "high"]


def _symptom_sets(results: list[dict[str, Any]]) -> tuple[list[set[str]], list[set[str]]]:
    y_true, y_pred = [], []
    for r in results:
        y_true.append(set(normalize_symptoms(r.get("true_symptoms", []))))
        out = r.get("output", {})
        y_pred.append(set(normalize_symptoms(out.get("symptoms", []))))
    return y_true, y_pred


def symptom_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    y_true, y_pred = _symptom_sets(results)
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        tp += len(t & p)
        fp += len(p - t)
        fn += len(t - p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"symptom_precision": precision, "symptom_recall": recall, "symptom_f1": f1}


def risk_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    y_true = [str(r.get("true_risk", "low")).lower() for r in results]
    y_pred = [str(r.get("output", {}).get("risk_level", "low")).lower() for r in results]

    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    macro_f1 = (
        f1_score(y_true, y_pred, labels=RISK_LABELS, average="macro", zero_division=0)
        if y_true
        else 0.0
    )

    high_true = [1 if t == "high" else 0 for t in y_true]
    high_pred = [1 if p == "high" else 0 for p in y_pred]
    high_recall = recall_score(high_true, high_pred, zero_division=0) if any(high_true) else 0.0

    return {
        "risk_accuracy": float(acc),
        "risk_macro_f1": float(macro_f1),
        "high_risk_recall": float(high_recall),
    }


def safety_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    n = len(results) or 1
    unsafe = sum(1 for r in results if r.get("unsafe_flag"))
    halluc = sum(1 for r in results if r.get("hallucination_flag"))
    return {
        "unsafe_flag_rate": unsafe / n,
        "hallucination_flag_rate": halluc / n,
    }


def rag_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    """RAG quality metrics for runs that use retrieval."""
    cfg = load_config()
    threshold = float(cfg.get("knowledge_base", {}).get("score_threshold", 0.35))
    rag_rows = [r for r in results if r.get("use_rag")]
    if not rag_rows:
        return {
            "retrieval_hit_rate": 0.0,
            "avg_retrieval_score": 0.0,
            "unsupported_evidence_rate": 0.0,
            "empty_retrieval_rate": 0.0,
        }

    hits = 0
    empty = 0
    unsupported = 0
    scores: list[float] = []

    for r in rag_rows:
        retrieval = r.get("retrieval") or {}
        chunks = retrieval.get("retrieved_chunks", [])
        if not chunks:
            empty += 1
            continue
        chunk_scores = [float(c.get("score", 0)) for c in chunks]
        max_score = max(chunk_scores) if chunk_scores else 0.0
        scores.extend(chunk_scores)
        if max_score >= threshold:
            hits += 1

        flags = r.get("output", {}).get("safety_flags", {})
        if isinstance(flags, dict) and flags.get("unsupported_diagnosis"):
            unsupported += 1
        elif max_score < threshold:
            unsupported += 1

    n = len(rag_rows)
    return {
        "retrieval_hit_rate": hits / n,
        "avg_retrieval_score": sum(scores) / len(scores) if scores else 0.0,
        "unsupported_evidence_rate": unsupported / n,
        "empty_retrieval_rate": empty / n,
    }


def runtime_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    n = len(results) or 1
    times = [float(r.get("runtime_sec", 0)) for r in results]
    json_ok = sum(1 for r in results if r.get("json_parse_ok"))
    return {
        "avg_runtime_sec": sum(times) / n,
        "total_runtime_sec": sum(times),
        "json_parse_success_rate": json_ok / n,
    }


def compute_run_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    m: dict[str, float] = {}
    m.update(symptom_metrics(results))
    m.update(risk_metrics(results))
    m.update(safety_metrics(results))
    m.update(runtime_metrics(results))
    if any(r.get("use_rag") for r in results):
        m.update(rag_metrics(results))
    return m


def evaluate_results(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_exp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_exp[r["experiment_name"]].append(r)

    all_metrics: dict[str, dict[str, float]] = {}
    for exp_name, rows in by_exp.items():
        all_metrics[exp_name] = compute_run_metrics(rows)
    return all_metrics


def build_run_record(
    run_cfg: RunConfig,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = compute_run_metrics(results)
    return {
        "experiment_name": run_cfg.experiment_name,
        "run_config": run_cfg,
        "metrics": metrics,
        "results": results,
    }


def save_metrics(metrics: dict[str, dict[str, float]], path: str | None = None) -> None:
    cfg = load_config()
    out_path = path or cfg["output"]["metrics_json"]
    ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def append_metrics_csv(run_records: list[dict[str, Any]], path: str | None = None) -> None:
    cfg = load_config()
    out_path = Path(path or cfg["output"]["metrics_csv"])
    ensure_dir(str(out_path))

    rows = []
    for rec in run_records:
        rc = rec.get("run_config")
        meta = rc.to_dict() if isinstance(rc, RunConfig) else rec.get("run_config", {})
        m = rec.get("metrics", {})
        rows.append(
            {
                "experiment_name": rec.get("experiment_name"),
                "phase": meta.get("phase"),
                "model": meta.get("model"),
                "structure": meta.get("structure"),
                "sample_size": meta.get("sample_size"),
                "use_rag": meta.get("use_rag"),
                "knowledge_base_mode": meta.get("knowledge_base_mode"),
                "ablation": meta.get("ablation"),
                "ensemble_method": meta.get("ensemble_method"),
                "weights": json.dumps(meta.get("weights")),
                "risk_accuracy": m.get("risk_accuracy"),
                "high_risk_recall": m.get("high_risk_recall"),
                "symptom_f1": m.get("symptom_f1"),
                "unsafe_flag_rate": m.get("unsafe_flag_rate"),
                "hallucination_flag_rate": m.get("hallucination_flag_rate"),
                "json_parse_success_rate": m.get("json_parse_success_rate"),
                "avg_runtime_sec": m.get("avg_runtime_sec"),
                "retrieval_hit_rate": m.get("retrieval_hit_rate"),
                "avg_retrieval_score": m.get("avg_retrieval_score"),
                "unsupported_evidence_rate": m.get("unsupported_evidence_rate"),
                "empty_retrieval_rate": m.get("empty_retrieval_rate"),
            }
        )

    df = pd.DataFrame(rows)
    if out_path.exists():
        old = pd.read_csv(out_path)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=["experiment_name"], keep="last"
        )
    df.to_csv(out_path, index=False)


def save_predictions_csv(results: list[dict[str, Any]], path: str | None = None) -> None:
    cfg = load_config()
    out_path = path or cfg["output"]["predictions_csv"]
    ensure_dir(out_path)
    rows = []
    for r in results:
        out = r.get("output", {})
        rc = r.get("run_config")
        meta = rc.to_dict() if isinstance(rc, RunConfig) else {}
        rows.append(
            {
                "case_id": r["case_id"],
                "experiment_name": r["experiment_name"],
                "phase": meta.get("phase"),
                "model": meta.get("model"),
                "structure": meta.get("structure"),
                "true_risk": r.get("true_risk"),
                "predicted_risk": out.get("risk_level"),
                "true_symptoms": ",".join(r.get("true_symptoms", []) or []),
                "predicted_symptoms": ",".join(out.get("symptoms", []) or []),
                "hallucination_flag": r.get("hallucination_flag"),
                "unsafe_flag": r.get("unsafe_flag"),
                "json_parse_ok": r.get("json_parse_ok"),
                "runtime_sec": r.get("runtime_sec"),
                "response_preview": (out.get("response") or "")[:200],
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def persist_run_to_db(
    run_cfg: RunConfig,
    metrics: dict[str, float],
    db: Database | None = None,
) -> None:
    db = db or Database()
    meta = run_cfg.to_dict()
    db.save_experiment_run(run_cfg.experiment_name, meta, metrics)
    for metric_name, metric_value in metrics.items():
        db.save_metric(run_cfg.experiment_name, metric_name, float(metric_value))


def finalize_run(
    run_cfg: RunConfig,
    results: list[dict[str, Any]],
    all_run_records: list[dict[str, Any]],
    db: Database | None = None,
) -> dict[str, Any]:
    metrics = compute_run_metrics(results)
    record = build_run_record(run_cfg, results)
    all_run_records.append(record)
    persist_run_to_db(run_cfg, metrics, db)
    append_metrics_csv([record])
    return record


def write_experiment_summary(
    run_records: list[dict[str, Any]],
    phase_state: dict[str, Any],
    path: str | None = None,
) -> None:
    cfg = load_config()
    out_path = Path(path or cfg["output"]["experiment_summary_md"])
    ensure_dir(str(out_path))

    best = pick_best(run_records)
    lines = [
        "# Experiment Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Sequential Experimental Design",
        "",
        "This project uses phased expansion with pruning (not exhaustive grid search).",
        "",
        "## Phase State",
        "",
        "```json",
        json.dumps(phase_state, indent=2),
        "```",
        "",
        "## Best Overall Run",
        "",
    ]
    if best:
        rc = best.get("run_config")
        meta = rc.to_dict() if isinstance(rc, RunConfig) else {}
        m = best.get("metrics", {})
        lines.extend(
            [
                f"- **Experiment:** `{best.get('experiment_name')}`",
                f"- **Phase:** {meta.get('phase')}",
                f"- **Model:** {meta.get('model')}",
                f"- **Structure:** {meta.get('structure')}",
                f"- **Sample size:** {meta.get('sample_size')}",
                f"- **High-risk recall:** {m.get('high_risk_recall', 0):.3f}",
                f"- **Unsafe rate:** {m.get('unsafe_flag_rate', 0):.3f}",
                f"- **Symptom F1:** {m.get('symptom_f1', 0):.3f}",
                f"- **JSON success:** {m.get('json_parse_success_rate', 0):.3f}",
                "",
            ]
        )
    else:
        lines.append("_No runs completed._\n")

    lines.extend(["## All Runs (ranked)", "", "| Rank | Experiment | Phase | Model | Structure | High-Risk Recall | Unsafe ↓ | Symptom F1 | JSON | Time(s) |", "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"])

    ranked = rank_runs(run_records, top_k=len(run_records))
    for i, rec in enumerate(ranked, 1):
        rc = rec.get("run_config")
        meta = rc.to_dict() if isinstance(rc, RunConfig) else {}
        m = rec.get("metrics", {})
        lines.append(
            f"| {i} | `{rec.get('experiment_name', '')[:40]}` | {meta.get('phase')} | {meta.get('model')} | {meta.get('structure')} | "
            f"{m.get('high_risk_recall', 0):.3f} | {m.get('unsafe_flag_rate', 0):.3f} | {m.get('symptom_f1', 0):.3f} | "
            f"{m.get('json_parse_success_rate', 0):.3f} | {m.get('avg_runtime_sec', 0):.1f} |"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_evaluation_pipeline(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metrics = evaluate_results(results)
    save_metrics(metrics)
    save_predictions_csv(results)
    db = Database()
    for exp_name, vals in metrics.items():
        for metric_name, metric_value in vals.items():
            db.save_metric(exp_name, metric_name, float(metric_value))
    return metrics
