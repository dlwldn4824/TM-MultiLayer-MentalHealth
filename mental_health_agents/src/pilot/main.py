"""CounselChat pilot benchmark entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from src.config import PROJECT_ROOT, load_config
from src.ollama_client import OllamaClient
from src.utils import ensure_dir

from src.pilot.evaluation import evaluate_all
from src.pilot.io import attach_retrieval_from_row
from src.pilot.runners import ALL_STRUCTURES, run_structure

logger = logging.getLogger(__name__)

CHECKPOINT_COLUMNS = [
    "case_id",
    "structure",
    "topic",
    "question_text",
    "reference_answer",
    "model_response",
    "runtime_sec",
    "retrieval_json",
    "agent_trace_json",
]


def _load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    df = pd.read_csv(path)
    for col in CHECKPOINT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def _done_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    if df.empty:
        return set()
    return {
        (str(row.case_id), str(row.structure))
        for row in df.itertuples(index=False)
    }


def _append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    line = pd.DataFrame([{k: row.get(k, "") for k in CHECKPOINT_COLUMNS}])
    header = not path.exists() or path.stat().st_size == 0
    line.to_csv(path, mode="a", header=header, index=False)


def _write_progress(
    progress_path: Path,
    *,
    cases_n: int,
    struct_list: list[str],
    done: set[tuple[str, str]],
    last_saved: tuple[str, str] | None = None,
) -> None:
    expected = cases_n * len(struct_list)
    by_structure: dict[str, int] = {}
    for s in struct_list:
        by_structure[s] = sum(1 for _, st in done if st == s)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "cases": cases_n,
        "structures": struct_list,
        "completed_pairs": len(done),
        "expected_pairs": expected,
        "percent": round(100.0 * len(done) / expected, 1) if expected else 0.0,
        "by_structure": by_structure,
        "last_saved": {"case_id": last_saved[0], "structure": last_saved[1]} if last_saved else None,
    }
    progress_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_pilot_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config_pilot.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _resolve_pilot_paths(cfg)


def _resolve_pilot_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    rag = cfg.get("rag", {})
    for key in ("chroma_dir",):
        if key in rag:
            p = Path(rag[key])
            if not p.is_absolute():
                rag[key] = str(PROJECT_ROOT / p)

    out = cfg.get("output", {})
    for key in out:
        p = Path(out[key])
        if not p.is_absolute():
            out[key] = str(PROJECT_ROOT / p)
    return cfg


def write_summary(
    path: str,
    metrics: dict[str, Any],
    cfg: dict[str, Any],
    n_cases: int,
) -> None:
    lines = [
        "# CounselChat Pilot Benchmark Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Setup",
        f"- Dataset: `{cfg['pilot']['dataset_id']}` (n={n_cases})",
        f"- Model: `{cfg['ollama']['model']}`",
        f"- KB sources: {', '.join(cfg['pilot']['source_ids'])}",
        "- Evaluation: BERTScore + ROUGE-L (reference), LLM Judge, Retrieval Judge",
        "- No pseudo-labels, no rule-based metrics",
        "",
        "## Results by structure",
        "",
        "| Structure | BERTScore F1 | ROUGE-L | Faithfulness | Relevancy | Empathy | Safety | Retrieval Rel. | Avg time (s) |",
        "|-----------|--------------|---------|--------------|-----------|---------|--------|----------------|--------------|",
    ]
    for name in cfg["pilot"].get("structures", ALL_STRUCTURES):
        m = metrics.get(name, {})
        lines.append(
            f"| {name} | {m.get('bertscore_f1', 0):.3f} | {m.get('rouge_l', 0):.3f} | "
            f"{m.get('judge_faithfulness', 0):.2f} | {m.get('judge_answer_relevancy', 0):.2f} | "
            f"{m.get('judge_empathy', 0):.2f} | {m.get('judge_safety', 0):.2f} | "
            f"{m.get('retrieval_relevance', 0):.2f} | {m.get('avg_runtime_sec', 0):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "- **BERTScore / ROUGE-L**: lexical-semantic overlap with CounselChat therapist `answerText` (reference).",
            "- **LLM Judge (0–5)**: faithfulness to reference tone/content, relevancy to client question, empathy, safety.",
            "- **Retrieval relevance**: LLM-rated usefulness of official KB chunks (RAG structures only).",
            "- Higher agent count → more LLM calls → longer runtime; compare trade-offs, not only quality.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_pilot(
    config_path: str | None = None,
    sample_size: int | None = None,
    structures: list[str] | None = None,
    skip_eval: bool = False,
    skip_llm_judge: bool = False,
    skip_retrieval_judge: bool = False,
    eval_only: bool = False,
    rebuild_kb: bool = True,
    resume: bool = True,
    fresh: bool = False,
) -> dict[str, Any]:
    cfg = load_pilot_config(config_path)
    pilot_cfg = cfg["pilot"]
    n = sample_size or pilot_cfg.get("sample_size", 30)
    struct_list = structures or pilot_cfg.get("structures", list(ALL_STRUCTURES))

    out = cfg["output"]
    ensure_dir(out["predictions_csv"])

    client = OllamaClient(cfg)
    all_rows: list[dict[str, Any]] = []

    if eval_only:
        pred_path = Path(out["predictions_csv"])
        if not pred_path.exists():
            raise FileNotFoundError(f"No predictions at {pred_path}; run inference first.")
        pred_df = pd.read_csv(pred_path)
        all_rows = pred_df.drop_duplicates(subset=["case_id", "structure"], keep="last").to_dict(
            orient="records"
        )
        for r in all_rows:
            attach_retrieval_from_row(r)
            if isinstance(r.get("model_response"), float) and math.isnan(r["model_response"]):
                r["model_response"] = ""
            if isinstance(r.get("reference_answer"), float) and math.isnan(r["reference_answer"]):
                r["reference_answer"] = ""
        cases_n = pred_df["case_id"].nunique()
        logger.info("Eval-only mode: %d rows from %s", len(all_rows), pred_path)
    else:
        from src.pilot.counsel_data import load_counsel_cases
        from src.pilot.kb import PilotRAG

        cases = load_counsel_cases(
            dataset_id=pilot_cfg["dataset_id"],
            split=pilot_cfg.get("dataset_split", "train"),
            sample_size=n,
            seed=pilot_cfg.get("random_seed", 42),
        )
        logger.info("Loaded %d CounselChat cases", len(cases))
        cases_n = len(cases)

        rag = PilotRAG(cfg)
        if rebuild_kb:
            rag.build_index(reset=True)
        elif rag._collection.count() == 0:
            rag.build_index(reset=True)

    if not eval_only:
        pred_path = Path(out["predictions_csv"])
        progress_path = Path(out.get("progress_json", str(pred_path.parent / "progress.json")))
        if fresh and pred_path.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            backup = pred_path.with_name(f"predictions_{ts}.csv.bak")
            pred_path.rename(backup)
            logger.info("Fresh run: archived predictions to %s", backup)
            if progress_path.exists():
                progress_path.unlink()
        checkpoint_df = _load_checkpoint(pred_path) if resume else pd.DataFrame(columns=CHECKPOINT_COLUMNS)
        done = _done_keys(checkpoint_df)
        if resume and done:
            logger.info("Resume: skipping %d existing (case_id, structure) pairs", len(done))
        _write_progress(progress_path, cases_n=cases_n, struct_list=struct_list, done=done)

        for structure in struct_list:
            use_rag = structure != "single"
            for case in tqdm(cases, desc=structure):
                cid = str(case["case_id"])
                if (cid, structure) in done:
                    continue
                try:
                    result = run_structure(
                        structure,
                        client,
                        rag if use_rag else None,
                        case,
                    )
                except Exception as e:
                    logger.exception("Failed %s case %s: %s", structure, case["case_id"], e)
                    result = {
                        "case_id": case["case_id"],
                        "structure": structure,
                        "question_text": case["question_text"],
                        "reference_answer": case["reference_answer"],
                        "response": f"[error: {e}]",
                        "retrieval": None,
                        "runtime_sec": 0.0,
                        "topic": case.get("topic", ""),
                    }

                retrieval = result.get("retrieval")
                row = {
                    "case_id": result["case_id"],
                    "structure": result["structure"],
                    "topic": result.get("topic", case.get("topic", "")),
                    "question_text": result["question_text"],
                    "reference_answer": result["reference_answer"],
                    "model_response": result.get("response", ""),
                    "runtime_sec": round(result.get("runtime_sec", 0.0), 2),
                    "retrieval": retrieval,
                    "retrieval_json": json.dumps(retrieval, ensure_ascii=False) if retrieval else "",
                    "agent_trace_json": json.dumps(
                        result.get("agent_trace", []), ensure_ascii=False
                    )[:8000],
                }
                save_row = {k: row[k] for k in CHECKPOINT_COLUMNS}
                _append_checkpoint(pred_path, save_row)
                done.add((cid, structure))
                _write_progress(
                    progress_path,
                    cases_n=cases_n,
                    struct_list=struct_list,
                    done=done,
                    last_saved=(cid, structure),
                )
                all_rows.append(row)

        checkpoint_df = _load_checkpoint(pred_path)
        all_rows = checkpoint_df.to_dict(orient="records")
        for r in all_rows:
            attach_retrieval_from_row(r)
        logger.info("Checkpoint: %d rows in %s", len(checkpoint_df), pred_path)

    metrics: dict[str, Any] = {}
    pred_path = Path(out["predictions_csv"])
    checkpoint_df = _load_checkpoint(pred_path) if pred_path.exists() else pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    done_pairs = _done_keys(checkpoint_df)

    if eval_only:
        inference_complete = len(checkpoint_df) > 0
        expected_pairs = len(checkpoint_df)
    else:
        sample_ids = {str(c["case_id"]) for c in cases}
        expected_pairs = len(sample_ids) * len(struct_list)
        inference_complete = all((cid, st) in done_pairs for cid in sample_ids for st in struct_list)

    if not skip_eval and not eval_only and not inference_complete:
        logger.warning(
            "Inference incomplete (%s/%s pairs). Skipping evaluation. "
            "Re-run: python scripts/run_pilot.py --no-rebuild-kb",
            len(done_pairs),
            expected_pairs,
        )
        skip_eval = True

    if not skip_eval:
        cfg_eval = dict(cfg)
        ev = dict(cfg.get("evaluation", {}))
        if skip_llm_judge:
            ev["run_llm_judge"] = False
        if skip_retrieval_judge:
            ev["run_retrieval_judge"] = False
        cfg_eval["evaluation"] = ev
        logger.info(
            "Starting evaluation on %d rows → %s",
            len(all_rows),
            out.get("metrics_json", "metrics.json"),
        )
        metrics, judge_df, retr_df = evaluate_all(all_rows, client, cfg_eval)
        Path(out["metrics_json"]).write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        judge_df.to_csv(out["judge_scores_csv"], index=False)
        retr_df.to_csv(out["retrieval_scores_csv"], index=False)
        write_summary(out["summary_md"], metrics, cfg, cases_n)
        logger.info("Wrote metrics, judge scores, retrieval scores, summary")

    return {"cases": cases_n, "predictions": len(all_rows), "metrics": metrics}


class _FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class _FlushingFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def _setup_pilot_logging(log_path: Path | None = None) -> None:
    import sys

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    console = _FlushingStreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = _FlushingFileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    logging.getLogger(__name__).info("Logging to stderr%s", f" and {log_path}" if log_path else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="CounselChat pilot benchmark")
    parser.add_argument("--config", default=None, help="Path to config_pilot.yaml")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--structures", nargs="+", default=None)
    parser.add_argument("--skip-eval", action="store_true", help="Inference only")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing predictions.csv")
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--skip-retrieval-judge", action="store_true")
    parser.add_argument("--no-rebuild-kb", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not load existing predictions.csv (still appends per case unless --fresh)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Archive predictions.csv and progress.json, start from scratch",
    )
    args = parser.parse_args()

    cfg_pre = load_pilot_config(args.config)
    log_file = Path(cfg_pre["output"].get("pilot_log", "outputs/pilot/pilot_run.log"))
    if not log_file.is_absolute():
        log_file = PROJECT_ROOT / log_file
    _setup_pilot_logging(log_file)

    run_pilot(
        config_path=args.config,
        sample_size=args.sample_size,
        structures=args.structures,
        skip_eval=args.skip_eval,
        skip_llm_judge=args.skip_llm_judge,
        skip_retrieval_judge=args.skip_retrieval_judge,
        eval_only=args.eval_only,
        rebuild_kb=not args.no_rebuild_kb,
        resume=not args.no_resume,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
