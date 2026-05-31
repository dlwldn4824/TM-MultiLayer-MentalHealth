"""Phase 3: Ensemble inference comparison (builds on Phase 2 CounselChat benchmark)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.config import PROJECT_ROOT
from src.database import Database
from src.model_benchmark import rank_combinations
from src.pilot.io import attach_retrieval_from_row
from src.response_ensemble import JUDGE_CRITERIA, apply_ensemble, model_performance_weights
from src.utils import ensure_dir

logger = logging.getLogger(__name__)

CHECKPOINT_COLUMNS = [
    "case_id",
    "ensemble_method",
    "structure",
    "selected_model",
    "topic",
    "question_text",
    "reference_answer",
    "model_response",
    "candidate_models",
    "ensemble_meta_json",
    "runtime_sec",
    "retrieval_json",
]


def _setup_logging(log_path: Path | None) -> None:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def _load_eval_config(config_name: str) -> dict[str, Any]:
    cfg_path = PROJECT_ROOT / config_name
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_phase3_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config_phase3.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _resolve_paths(cfg)


def _resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    p3 = cfg.get("phase3", {})
    for key in ("phase2_predictions_csv", "phase2_metrics_json", "phase2_judge_scores_csv"):
        if key in p3:
            p = Path(p3[key])
            if not p.is_absolute():
                p3[key] = str(PROJECT_ROOT / p)
    out = cfg.get("output", {})
    for key in out:
        p = Path(out[key])
        if not p.is_absolute():
            out[key] = str(PROJECT_ROOT / p)
    db = cfg.get("output_db", {})
    if "sqlite_path" in db:
        p = Path(db["sqlite_path"])
        if not p.is_absolute():
            db["sqlite_path"] = str(PROJECT_ROOT / p)
    return cfg


def _load_phase2_matrix(metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.exists():
        return pd.DataFrame()
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []
    for key, m in data.items():
        if "|" in key:
            model, structure = key.split("|", 1)
        else:
            model, structure = m.get("model", ""), m.get("structure", key)
        rows.append({"model": model, "structure": structure, **m})
    return pd.DataFrame(rows)


def _select_phase2_winners(
    matrix: pd.DataFrame,
    *,
    top_k_models: int,
    structure: str | None,
) -> tuple[str, list[str]]:
    if matrix.empty:
        raise FileNotFoundError("Phase 2 metrics not found. Run Phase 2 first or pass --models.")

    ranked = rank_combinations(matrix)
    if structure:
        ranked = ranked[ranked["structure"] == structure]
    if ranked.empty:
        raise ValueError(f"No Phase 2 rows for structure={structure!r}")

    best_structure = str(ranked.iloc[0]["structure"])
    subset = ranked[ranked["structure"] == best_structure]
    models: list[str] = []
    for model in subset["model"]:
        if model not in models:
            models.append(str(model))
        if len(models) >= top_k_models:
            break
    return best_structure, models


def _load_judge_lookup(judge_path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    if not judge_path.exists():
        return {}
    df = pd.read_csv(judge_path)
    lookup: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in df.itertuples(index=False):
        key = (str(row.case_id), str(getattr(row, "model", "")), str(row.structure))
        lookup[key] = {k: float(getattr(row, k, 0.0)) for k in JUDGE_CRITERIA if hasattr(row, k)}
    return lookup


def _build_case_groups(
    predictions: pd.DataFrame,
    *,
    structure: str,
    models: list[str],
) -> dict[str, list[dict[str, Any]]]:
    filtered = predictions[
        (predictions["structure"] == structure) & (predictions["model"].isin(models))
    ].drop_duplicates(subset=["case_id", "model"], keep="last")

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in filtered.to_dict(orient="records"):
        cid = str(row["case_id"])
        groups.setdefault(cid, []).append(row)
    return groups


def _ensemble_rows_for_method(
    *,
    method: str,
    structure: str,
    models: list[str],
    case_groups: dict[str, list[dict[str, Any]]],
    judge_lookup: dict[tuple[str, str, str], dict[str, float]],
    model_weights: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id, items in sorted(case_groups.items()):
        candidates = {str(r["model"]): str(r.get("model_response", "")) for r in items}
        if len(candidates) < 2:
            continue

        judge_by_model: dict[str, dict[str, float]] = {}
        for model in candidates:
            key = (case_id, model, structure)
            judge_by_model[model] = judge_lookup.get(key, {k: 0.0 for k in JUDGE_CRITERIA})

        response, selected_model, meta = apply_ensemble(
            method,
            candidates,
            judge_by_model,
            model_weights=model_weights,
        )
        base = items[0]
        retrieval = base.get("retrieval")
        if not isinstance(retrieval, dict):
            retrieval = None

        rows.append(
            {
                "case_id": case_id,
                "ensemble_method": method,
                "structure": structure,
                "selected_model": selected_model,
                "topic": base.get("topic", ""),
                "question_text": base.get("question_text", ""),
                "reference_answer": base.get("reference_answer", ""),
                "model_response": response,
                "candidate_models": ",".join(sorted(candidates)),
                "ensemble_meta_json": json.dumps(meta, ensure_ascii=False),
                "runtime_sec": sum(float(r.get("runtime_sec") or 0.0) for r in items),
                "retrieval": retrieval,
                "retrieval_json": base.get("retrieval_json", ""),
                "model": f"ensemble:{method}",
            }
        )
    return rows


def _write_ensemble_summary(
    path: Path,
    cfg: dict[str, Any],
    *,
    structure: str,
    models: list[str],
    metrics: dict[str, Any],
    matrix: pd.DataFrame,
    ranked: pd.DataFrame,
    single_best: dict[str, float] | None,
) -> None:
    lines = [
        "# Phase 3: Ensemble Comparison Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Setup",
        f"- Base structure: `{structure}`",
        f"- Candidate models: {', '.join(models)}",
        f"- Ensemble methods: {', '.join(cfg['phase3']['ensemble_methods'])}",
        "- Evaluation: BERTScore, ROUGE-L, Faithfulness, Relevancy, Empathy, Safety, Runtime",
        "",
        "## Ensemble Results",
        "",
        _ensemble_matrix_md(matrix),
        "",
        "## Ranking",
        "",
        _ensemble_ranking_md(ranked),
        "",
        "## Analysis Questions",
        "",
    ]

    if ranked.empty:
        lines.append("_No ensemble results yet._")
    else:
        best = ranked.iloc[0]
        best_method = best["ensemble_method"]
        best_faith = best.get("judge_faithfulness", 0.0)
        best_safe = best.get("judge_safety", 0.0)
        best_runtime = best.get("avg_runtime_sec", 0.0)

        q1 = "Insufficient comparison data."
        if single_best:
            sb_score = (
                single_best.get("judge_faithfulness", 0.0) * 5
                + single_best.get("judge_safety", 0.0) * 4
                + single_best.get("judge_empathy", 0.0) * 3
            )
            ens_score = (
                best.get("judge_faithfulness", 0.0) * 5
                + best.get("judge_safety", 0.0) * 4
                + best.get("judge_empathy", 0.0) * 3
            )
            q1 = (
                f"Best ensemble `{best_method}` weighted score={ens_score:.2f} vs "
                f"single best model={sb_score:.2f}. "
                f"{'Ensemble wins' if ens_score > sb_score else 'Single model still leads'}."
            )

        q2 = (
            f"Best ensemble `{best_method}`: faithfulness={best_faith:.2f}, safety={best_safe:.2f}. "
            "Compare variance in `judge_scores.csv` for faithfulness/safety spread."
        )
        q3 = (
            f"Best ensemble avg runtime={best_runtime:.1f}s (sum of candidate runtimes per case). "
            "Compare against single-model runtime in Phase 2 outputs."
        )

        lines.extend(
            [
                "### 1. Is ensemble consistently better than single best model?",
                q1,
                "",
                "### 2. Can ensemble improve faithfulness and safety together?",
                q2,
                "",
                "### 3. Is runtime increase acceptable?",
                q3,
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def _ensemble_matrix_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No results yet._"
    headers = [
        "Method",
        "BERTScore",
        "ROUGE-L",
        "Faith",
        "Relev",
        "Empathy",
        "Safety",
        "Runtime(s)",
        "N",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['ensemble_method']} | {r['bertscore_f1']:.3f} | {r['rouge_l']:.3f} | "
            f"{r['judge_faithfulness']:.2f} | {r['judge_answer_relevancy']:.2f} | "
            f"{r['judge_empathy']:.2f} | {r['judge_safety']:.2f} | "
            f"{r['avg_runtime_sec']:.1f} | {int(r.get('n', 0))} |"
        )
    return "\n".join(lines)


def _ensemble_ranking_md(ranked: pd.DataFrame) -> str:
    if ranked.empty:
        return "_No results yet._"
    lines = [
        "| Rank | Method | Score | Faith | Safety | Empathy | Relev | Runtime(s) |",
        "|------|--------|-------|-------|--------|---------|-------|------------|",
    ]
    for _, r in ranked.iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['ensemble_method']} | {r['rank_score']:.2f} | "
            f"{r['judge_faithfulness']:.2f} | {r['judge_safety']:.2f} | "
            f"{r['judge_empathy']:.2f} | {r['judge_answer_relevancy']:.2f} | "
            f"{r['avg_runtime_sec']:.1f} |"
        )
    return "\n".join(lines)


def _export_phase3_outputs(
    out: dict[str, str],
    *,
    metrics: dict[str, Any],
    matrix: pd.DataFrame,
    ranked: pd.DataFrame,
    predictions: pd.DataFrame,
    judge_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
) -> None:
    ensure_dir(out["dir"])
    Path(out["metrics_json"]).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    matrix.to_csv(out["metrics_csv"], index=False)
    predictions.to_csv(out["predictions_csv"], index=False)
    ranked.to_csv(out["ensemble_results_csv"], index=False)
    if not judge_df.empty:
        judge_df.to_csv(out["judge_scores_csv"], index=False)
    if not runtime_df.empty:
        runtime_df.to_csv(out["runtime_metrics_csv"], index=False)


def run_phase3(
    config_path: str | None = None,
    *,
    models: list[str] | None = None,
    structure: str | None = None,
    methods: list[str] | None = None,
    skip_eval: bool = False,
    skip_llm_judge: bool = False,
    skip_retrieval_judge: bool = False,
    eval_only: bool = False,
) -> dict[str, Any]:
    cfg = load_phase3_config(config_path)
    p3 = cfg["phase3"]
    eval_cfg = _load_eval_config(p3.get("phase2_config", "config_phase2.yaml"))

    pred_path = Path(p3["phase2_predictions_csv"])
    metrics_path = Path(p3["phase2_metrics_json"])
    judge_path = Path(p3["phase2_judge_scores_csv"])

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Phase 2 predictions missing at {pred_path}. Run: python scripts/run_phase2.py"
        )

    predictions = pd.read_csv(pred_path)
    matrix = _load_phase2_matrix(metrics_path)
    top_k = p3.get("top_k_models", 3)
    chosen_structure, chosen_models = _select_phase2_winners(
        matrix,
        top_k_models=top_k,
        structure=structure or p3.get("structure"),
    )
    if models:
        chosen_models = models

    logger.info(
        "Phase 3 candidates: structure=%s models=%s",
        chosen_structure,
        chosen_models,
    )

    model_metrics = {
        str(r["model"]): r.to_dict()
        for _, r in matrix[matrix["structure"] == chosen_structure].iterrows()
        if str(r["model"]) in chosen_models
    }
    model_weights = model_performance_weights(model_metrics, chosen_models)
    logger.info("Model weights (Phase 2 performance): %s", model_weights)

    judge_lookup = _load_judge_lookup(judge_path)
    if not judge_lookup:
        logger.warning("No Phase 2 judge scores at %s; ensemble selection uses zero scores.", judge_path)

    case_groups = _build_case_groups(predictions, structure=chosen_structure, models=chosen_models)
    if not case_groups:
        raise ValueError(
            f"No overlapping Phase 2 predictions for structure={chosen_structure} models={chosen_models}"
        )

    ensemble_methods = methods or p3.get("ensemble_methods", [])
    all_ensemble_rows: list[dict[str, Any]] = []
    for method in ensemble_methods:
        logger.info("Building ensemble rows: %s", method)
        rows = _ensemble_rows_for_method(
            method=method,
            structure=chosen_structure,
            models=chosen_models,
            case_groups=case_groups,
            judge_lookup=judge_lookup,
            model_weights=model_weights,
        )
        logger.info("  %s: %d cases", method, len(rows))
        all_ensemble_rows.extend(rows)

    pred_out = Path(cfg["output"]["predictions_csv"])
    pd.DataFrame(all_ensemble_rows).to_csv(pred_out, index=False)
    logger.info("Wrote %d ensemble prediction rows to %s", len(all_ensemble_rows), pred_out)

    metrics: dict[str, Any] = {}
    all_judge: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []

    if not skip_eval and all_ensemble_rows:
        from src.ollama_client import OllamaClient
        from src.pilot.evaluation import evaluate_all

        cfg_eval = dict(eval_cfg)
        cfg_eval.setdefault("evaluation", {}).update(cfg.get("evaluation", {}))
        ev = dict(cfg_eval.get("evaluation", {}))
        if skip_llm_judge:
            ev["run_llm_judge"] = False
        if skip_retrieval_judge:
            ev["run_retrieval_judge"] = False
        cfg_eval["evaluation"] = ev

        judge_client = OllamaClient(cfg_eval)
        for method in ensemble_methods:
            subset = [r for r in all_ensemble_rows if r["ensemble_method"] == method]
            if not subset:
                continue
            for r in subset:
                attach_retrieval_from_row(r)

            eval_rows = [
                {
                    **r,
                    "structure": chosen_structure,
                    "model_response": r.get("model_response", ""),
                }
                for r in subset
            ]
            logger.info("Evaluating ensemble method %s (%d rows)", method, len(eval_rows))
            metrics_map, judge_df, retr_df = evaluate_all(eval_rows, judge_client, cfg_eval)
            agg = metrics_map.get(chosen_structure, {})
            agg["ensemble_method"] = method
            agg["structure"] = chosen_structure
            agg["selected_models"] = chosen_models
            metrics[method] = agg

            if not judge_df.empty:
                judge_df = judge_df.copy()
                judge_df["ensemble_method"] = method
                all_judge.append(judge_df)

            runtime_rows.append(
                {
                    "ensemble_method": method,
                    "structure": chosen_structure,
                    "avg_runtime_sec": agg.get("avg_runtime_sec"),
                    "n": agg.get("n"),
                }
            )

            db_path = cfg.get("output_db", {}).get("sqlite_path")
            if db_path:
                db = Database(db_path)
                db.save_experiment_run(
                    f"phase3__{method}",
                    {
                        "phase": "phase3",
                        "model": f"ensemble:{method}",
                        "structure": chosen_structure,
                        "sample_size": agg.get("n"),
                        "ensemble_method": method,
                    },
                    {k: float(v) for k, v in agg.items() if isinstance(v, (int, float))},
                )

    flat_metrics = {
        method: {**m, "ensemble_method": method}
        for method, m in metrics.items()
    }
    matrix_df = pd.DataFrame(
        [
            {
                "ensemble_method": method,
                "bertscore_f1": m.get("bertscore_f1", 0.0),
                "rouge_l": m.get("rouge_l", 0.0),
                "judge_faithfulness": m.get("judge_faithfulness", 0.0),
                "judge_answer_relevancy": m.get("judge_answer_relevancy", 0.0),
                "judge_empathy": m.get("judge_empathy", 0.0),
                "judge_safety": m.get("judge_safety", 0.0),
                "avg_runtime_sec": m.get("avg_runtime_sec", 0.0),
                "n": m.get("n", 0),
            }
            for method, m in flat_metrics.items()
        ]
    )

    ranked = rank_combinations(
        matrix_df.rename(columns={"ensemble_method": "model"}).assign(structure=chosen_structure)
    )
    if not ranked.empty and "model" in ranked.columns:
        ranked = ranked.rename(columns={"model": "ensemble_method"})

    single_best = None
    if not matrix.empty:
        subset = matrix[matrix["structure"] == chosen_structure]
        if not subset.empty:
            best_model = rank_combinations(subset).iloc[0]
            single_best = best_model.to_dict()

    out = cfg["output"]
    _export_phase3_outputs(
        out,
        metrics=metrics,
        matrix=matrix_df,
        ranked=ranked,
        predictions=pd.DataFrame(all_ensemble_rows),
        judge_df=pd.concat(all_judge, ignore_index=True) if all_judge else pd.DataFrame(),
        runtime_df=pd.DataFrame(runtime_rows),
    )
    _write_ensemble_summary(
        Path(out["summary_md"]),
        cfg,
        structure=chosen_structure,
        models=chosen_models,
        metrics=metrics,
        matrix=matrix_df,
        ranked=ranked,
        single_best=single_best,
    )

    print(f"Phase3 complete. Methods={ensemble_methods} cases={len(case_groups)}")
    if not ranked.empty:
        best = ranked.iloc[0]
        print(
            f"Best ensemble: {best['ensemble_method']} "
            f"(faith={best.get('judge_faithfulness', 0):.2f}, safety={best.get('judge_safety', 0):.2f})"
        )

    return {
        "structure": chosen_structure,
        "models": chosen_models,
        "cases": len(case_groups),
        "metrics": metrics,
        "ranked": ranked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Ensemble inference comparison")
    parser.add_argument("--config", default=None, help="Path to config_phase3.yaml")
    parser.add_argument("--models", default=None, help="Comma-separated candidate models")
    parser.add_argument("--structure", default=None, help="Override Phase 2 structure")
    parser.add_argument("--methods", default=None, help="Comma-separated ensemble methods")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Build ensembles from Phase 2 only")
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--skip-retrieval-judge", action="store_true")
    args = parser.parse_args()

    cfg_pre = load_phase3_config(args.config)
    log_file = Path(cfg_pre["output"].get("log", "outputs/phase3/phase3_run.log"))
    _setup_logging(log_file)

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else None

    run_phase3(
        config_path=args.config,
        models=models,
        structure=args.structure,
        methods=methods,
        skip_eval=args.skip_eval,
        skip_llm_judge=args.skip_llm_judge,
        skip_retrieval_judge=args.skip_retrieval_judge,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
