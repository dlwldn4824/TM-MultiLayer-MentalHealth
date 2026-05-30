"""Phase 2 CounselChat model comparison runner (reuses pilot inference + evaluation)."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from src.config import PROJECT_ROOT
from src.database import Database
from src.model_benchmark import (
    build_comparison_matrix,
    console_summary,
    experiment_key,
    export_outputs,
    rank_combinations,
    write_summary_md,
)
from src.ollama_client import OllamaClient
from src.pilot.counsel_data import load_counsel_cases
from src.pilot.evaluation import evaluate_all
from src.pilot.io import attach_retrieval_from_row
from src.pilot.kb import PilotRAG
from src.pilot.runners import run_structure
from src.utils import ensure_dir

logger = logging.getLogger(__name__)

CHECKPOINT_COLUMNS = [
    "case_id",
    "model",
    "structure",
    "topic",
    "question_text",
    "reference_answer",
    "model_response",
    "runtime_sec",
    "retrieval_json",
    "agent_trace_json",
    "prompt_tokens",
    "completion_tokens",
    "json_success_rate",
    "inference_ok",
]


class _StatsClient:
    """Wrap OllamaClient to track JSON parse success and token counts."""

    def __init__(self, client: OllamaClient):
        self._client = client
        self.llm_calls = 0
        self.json_ok = 0
        self.inference_errors = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def temperature(self) -> float:
        return self._client.temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._client.temperature = value

    def chat_json(self, user_prompt: str, system_prompt: str) -> dict[str, Any]:
        self.llm_calls += 1
        out = self._client.chat_json(user_prompt, system_prompt)
        if out.get("success") and out.get("parsed_json"):
            self.json_ok += 1
        if not out.get("success"):
            self.inference_errors += 1
        self.prompt_tokens += int(out.get("prompt_tokens") or 0)
        self.completion_tokens += int(out.get("completion_tokens") or 0)
        return out

    def generate(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        return self.chat_json(prompt, system or "")

    def with_model(self, model: str) -> _StatsClient:
        cloned = copy.copy(self)
        cloned._client = self._client.with_model(model)
        return cloned

    def reset(self) -> None:
        self.llm_calls = 0
        self.json_ok = 0
        self.inference_errors = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def json_success_rate(self) -> float:
        return self.json_ok / self.llm_calls if self.llm_calls else 0.0

    @property
    def token_throughput(self) -> float:
        total = self.prompt_tokens + self.completion_tokens
        return total  # per-case rate computed with runtime in runner


def load_phase2_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config_phase2.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _resolve_paths(cfg)


def _resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
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
    db = cfg.get("output_db", {})
    if "sqlite_path" in db:
        p = Path(db["sqlite_path"])
        if not p.is_absolute():
            db["sqlite_path"] = str(PROJECT_ROOT / p)
    return cfg


class _FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class _FlushingFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def _setup_logging(log_path: Path | None) -> None:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    sh = _FlushingStreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = _FlushingFileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def _load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    df = pd.read_csv(path)
    for col in CHECKPOINT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def _done_keys(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    if df.empty:
        return set()
    return {
        (str(r.case_id), str(r.model), str(r.structure))
        for r in df.itertuples(index=False)
    }


def _append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    line = pd.DataFrame([{k: row.get(k, "") for k in CHECKPOINT_COLUMNS}])
    header = not path.exists() or path.stat().st_size == 0
    line.to_csv(path, mode="a", header=header, index=False)


def _expected_triples(cases_n: int, models: list[str], structures: list[str]) -> int:
    return cases_n * len(models) * len(structures)


def _progress_percent(done: set[tuple[str, str, str]], expected: int) -> float:
    return round(100.0 * len(done) / expected, 1) if expected else 0.0


def _write_progress(
    progress_path: Path,
    *,
    models: list[str],
    structures: list[str],
    cases_n: int,
    done: set[tuple[str, str, str]],
    current: tuple[str, str] | None = None,
    last_case_id: str | None = None,
) -> dict[str, Any]:
    expected = _expected_triples(cases_n, models, structures)
    completed = len(done)
    percent = _progress_percent(done, expected)
    payload: dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "phase2",
        "cases": cases_n,
        "models": models,
        "structures": structures,
        "completed_triples": completed,
        "expected_triples": expected,
        "percent": percent,
        "current": {"model": current[0], "structure": current[1]} if current else None,
    }
    if last_case_id is not None:
        payload["last_case_id"] = last_case_id
    progress_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _log_progress(
    *,
    done: set[tuple[str, str, str]],
    expected: int,
    model: str | None = None,
    structure: str | None = None,
    case_id: str | None = None,
) -> None:
    completed = len(done)
    percent = _progress_percent(done, expected)
    parts = [f"Progress: {completed}/{expected} ({percent}%)"]
    if model and structure:
        parts.append(f"{model} / {structure}")
    if case_id:
        parts.append(f"case_id={case_id}")
    logger.info(" | ".join(parts))


def _sync_progress(
    progress_path: Path,
    *,
    models: list[str],
    structures: list[str],
    cases_n: int,
    done: set[tuple[str, str, str]],
    expected: int,
    current: tuple[str, str] | None = None,
    case_id: str | None = None,
    log: bool = True,
) -> None:
    _write_progress(
        progress_path,
        models=models,
        structures=structures,
        cases_n=cases_n,
        done=done,
        current=current,
        last_case_id=case_id,
    )
    if log:
        _log_progress(
            done=done,
            expected=expected,
            model=current[0] if current else None,
            structure=current[1] if current else None,
            case_id=case_id,
        )


def _evaluate_combo(
    rows: list[dict[str, Any]],
    client: OllamaClient,
    cfg: dict[str, Any],
    model: str,
    structure: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    subset = [r for r in rows if r.get("model") == model and r.get("structure") == structure]
    if not subset:
        return {}
    metrics_map, judge_df, retr_df = evaluate_all(subset, client, cfg)
    m = metrics_map.get(structure, {})
    m["model"] = model
    m["structure"] = structure

    runtimes = [float(r.get("runtime_sec") or 0) for r in subset]
    json_rates = [float(r.get("json_success_rate") or 0) for r in subset]
    throughputs = [float(r.get("token_throughput") or 0) for r in subset]
    stable = [int(r.get("inference_ok") or 0) for r in subset]

    m["json_success_rate"] = sum(json_rates) / len(json_rates) if json_rates else 0.0
    m["token_throughput"] = sum(throughputs) / len(throughputs) if throughputs else 0.0
    m["inference_stability"] = sum(stable) / len(stable) if stable else 0.0
    m["avg_runtime_sec"] = sum(runtimes) / len(runtimes) if runtimes else 0.0
    return m, judge_df, retr_df


def _persist_sqlite(
    db: Database,
    model: str,
    structure: str,
    agg: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    exp_key = experiment_key(model, structure)
    db.save_experiment_run(
        exp_key,
        {
            "phase": "phase2",
            "model": model,
            "structure": structure,
            "sample_size": agg.get("n"),
            "use_rag": structure != "single",
            "dataset_name": "nbertagnolli/counsel-chat",
        },
        {k: float(v) for k, v in agg.items() if isinstance(v, (int, float))},
    )
    for r in rows:
        if r.get("model") != model or r.get("structure") != structure:
            continue
        db.save_phase2_counsel_row(
            {
                "phase": "phase2",
                "model": model,
                "structure": structure,
                "case_id": str(r.get("case_id")),
                "runtime_sec": r.get("runtime_sec"),
                "experiment_key": exp_key,
                "json_success_rate": r.get("json_success_rate"),
                "token_throughput": r.get("token_throughput"),
                "inference_success": r.get("inference_ok"),
                "metrics_json": {},
            }
        )


def run_phase2(
    config_path: str | None = None,
    sample_size: int | None = None,
    models: list[str] | None = None,
    structures: list[str] | None = None,
    skip_eval: bool = False,
    skip_llm_judge: bool = False,
    skip_retrieval_judge: bool = False,
    eval_only: bool = False,
    rebuild_kb: bool = False,
    resume: bool = True,
    fresh: bool = False,
) -> dict[str, Any]:
    cfg = load_phase2_config(config_path)
    p2 = cfg["phase2"]
    n = sample_size or p2.get("sample_size", 30)
    model_list = models or p2.get("models", [])
    struct_list = structures or p2.get("structures", ["single_rag", "three_agent"])
    out = cfg["output"]
    ensure_dir(out["dir"])

    judge_client = OllamaClient(cfg)
    all_rows: list[dict[str, Any]] = []
    cases_n = 0

    pred_path = Path(out["predictions_csv"])
    progress_path = Path(out.get("progress_json", str(Path(out["dir"]) / "progress.json")))

    if fresh and pred_path.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        pred_path.rename(pred_path.with_name(f"predictions_{ts}.csv.bak"))
        if progress_path.exists():
            progress_path.unlink()

    if eval_only:
        if not pred_path.exists():
            raise FileNotFoundError(f"No predictions at {pred_path}")
        df = pd.read_csv(pred_path).drop_duplicates(
            subset=["case_id", "model", "structure"], keep="last"
        )
        all_rows = df.to_dict(orient="records")
        for r in all_rows:
            attach_retrieval_from_row(r)
        cases_n = len({r["case_id"] for r in all_rows})
        model_list = sorted({str(r["model"]) for r in all_rows if r.get("model")})
        struct_list = sorted({str(r["structure"]) for r in all_rows if r.get("structure")})
        logger.info("Eval-only: %d rows from %s", len(all_rows), pred_path)
    else:
        cases = load_counsel_cases(
            dataset_id=p2["dataset_id"],
            split=p2.get("dataset_split", "train"),
            sample_size=n,
            seed=p2.get("random_seed", 42),
        )
        cases_n = len(cases)
        logger.info("Loaded %d CounselChat cases", cases_n)

        rag = PilotRAG(cfg)
        if rebuild_kb or rag._collection.count() == 0:
            rag.build_index(reset=rebuild_kb)
        else:
            logger.info("Reusing KB index (%d chunks)", rag._collection.count())

        checkpoint_df = _load_checkpoint(pred_path) if resume else pd.DataFrame(columns=CHECKPOINT_COLUMNS)
        done = _done_keys(checkpoint_df)
        expected_triples = _expected_triples(cases_n, model_list, struct_list)
        logger.info(
            "Phase2 grid: %d triples (%d models × %d structures × %d cases)",
            expected_triples,
            len(model_list),
            len(struct_list),
            cases_n,
        )
        if resume and done:
            _log_progress(done=done, expected=expected_triples)
            logger.info("Resume: skipping %d existing (case_id, model, structure) triples", len(done))

        for model in model_list:
            base_client = OllamaClient(cfg).with_model(model)
            logger.info("=== Model: %s ===", model)
            for structure in struct_list:
                logger.info("--- Structure: %s ---", structure)
                _sync_progress(
                    progress_path,
                    models=model_list,
                    structures=struct_list,
                    cases_n=cases_n,
                    done=done,
                    expected=expected_triples,
                    current=(model, structure),
                    log=True,
                )
                use_rag = structure != "single"
                pbar = tqdm(cases, desc=f"{model}/{structure}")
                for case in pbar:
                    cid = str(case["case_id"])
                    if (cid, model, structure) in done:
                        continue
                    stats = _StatsClient(base_client)
                    inference_ok = 1
                    try:
                        result = run_structure(
                            structure,
                            stats,  # type: ignore[arg-type]
                            rag if use_rag else None,
                            case,
                        )
                    except Exception as e:
                        logger.exception("Failed %s %s case %s: %s", model, structure, cid, e)
                        inference_ok = 0
                        result = {
                            "case_id": case["case_id"],
                            "structure": structure,
                            "question_text": case["question_text"],
                            "reference_answer": case["reference_answer"],
                            "response": f"[error: {e}]",
                            "retrieval": None,
                            "runtime_sec": 0.0,
                            "topic": case.get("topic", ""),
                            "agent_trace": [],
                        }

                    runtime = float(result.get("runtime_sec") or 0.0)
                    total_tokens = stats.prompt_tokens + stats.completion_tokens
                    throughput = total_tokens / runtime if runtime > 0 else 0.0

                    retrieval = result.get("retrieval")
                    row = {
                        "case_id": result["case_id"],
                        "model": model,
                        "structure": structure,
                        "topic": result.get("topic", ""),
                        "question_text": result["question_text"],
                        "reference_answer": result["reference_answer"],
                        "model_response": result.get("response", ""),
                        "runtime_sec": round(runtime, 2),
                        "retrieval": retrieval,
                        "retrieval_json": json.dumps(retrieval, ensure_ascii=False) if retrieval else "",
                        "agent_trace_json": json.dumps(
                            result.get("agent_trace", []), ensure_ascii=False
                        )[:8000],
                        "prompt_tokens": stats.prompt_tokens,
                        "completion_tokens": stats.completion_tokens,
                        "json_success_rate": round(stats.json_success_rate, 4),
                        "token_throughput": round(throughput, 2),
                        "inference_ok": inference_ok,
                    }
                    save_row = {k: row[k] for k in CHECKPOINT_COLUMNS}
                    _append_checkpoint(pred_path, save_row)
                    done.add((cid, model, structure))
                    all_rows.append(row)
                    pct = _progress_percent(done, expected_triples)
                    pbar.set_postfix_str(f"overall {pct}%", refresh=False)
                    _sync_progress(
                        progress_path,
                        models=model_list,
                        structures=struct_list,
                        cases_n=cases_n,
                        done=done,
                        expected=expected_triples,
                        current=(model, structure),
                        case_id=cid,
                        log=True,
                    )

        _log_progress(done=done, expected=expected_triples)
        logger.info("Inference phase complete")
        checkpoint_df = _load_checkpoint(pred_path)
        all_rows = checkpoint_df.drop_duplicates(
            subset=["case_id", "model", "structure"], keep="last"
        ).to_dict(orient="records")
        for r in all_rows:
            attach_retrieval_from_row(r)
        logger.info("Checkpoint: %d rows in %s", len(all_rows), pred_path)

    metrics: dict[str, Any] = {}
    all_judge: list[pd.DataFrame] = []
    all_retr: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []

    if not skip_eval and all_rows:
        cfg_eval = dict(cfg)
        ev = dict(cfg.get("evaluation", {}))
        if skip_llm_judge:
            ev["run_llm_judge"] = False
        if skip_retrieval_judge:
            ev["run_retrieval_judge"] = False
        cfg_eval["evaluation"] = ev

        for model in model_list:
            for structure in struct_list:
                combo_rows = [
                    r
                    for r in all_rows
                    if r.get("model") == model and r.get("structure") == structure
                ]
                if not combo_rows:
                    continue
                logger.info("Evaluating %s / %s (%d rows)", model, structure, len(combo_rows))
                eval_rows = [{**r, "model_response": r.get("model_response", "")} for r in combo_rows]
                result = _evaluate_combo(eval_rows, judge_client, cfg_eval, model, structure)
                if not result:
                    continue
                agg, judge_df, retr_df = result
                key = f"{model}|{structure}"
                metrics[key] = agg
                if not judge_df.empty:
                    judge_df = judge_df.copy()
                    judge_df["model"] = model
                    all_judge.append(judge_df)
                if not retr_df.empty:
                    retr_df = retr_df.copy()
                    retr_df["model"] = model
                    all_retr.append(retr_df)
                runtime_rows.append(
                    {
                        "model": model,
                        "structure": structure,
                        "avg_runtime_sec": agg.get("avg_runtime_sec"),
                        "token_throughput": agg.get("token_throughput"),
                        "inference_stability": agg.get("inference_stability"),
                        "json_success_rate": agg.get("json_success_rate"),
                        "n": agg.get("n"),
                    }
                )
                db_path = cfg.get("output_db", {}).get("sqlite_path")
                if db_path:
                    _persist_sqlite(Database(db_path), model, structure, agg, combo_rows)

    df = build_comparison_matrix(metrics)
    ranked = rank_combinations(df)
    judge_out = pd.concat(all_judge, ignore_index=True) if all_judge else pd.DataFrame()
    retr_out = pd.concat(all_retr, ignore_index=True) if all_retr else pd.DataFrame()
    runtime_out = pd.DataFrame(runtime_rows)

    export_outputs(out, metrics, df, ranked, judge_out, retr_out, runtime_out)
    write_summary_md(out["summary_md"], cfg, metrics, df, ranked)

    summary_text = console_summary(df, ranked)
    print(summary_text)

    return {
        "cases": cases_n,
        "predictions": len(all_rows),
        "metrics": metrics,
        "matrix": df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: CounselChat model comparison")
    parser.add_argument("--config", default=None, help="Path to config_phase2.yaml")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model tags (default: config order)",
    )
    parser.add_argument(
        "--structures",
        default=None,
        help="Comma-separated structures (default: single_rag,three_agent)",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--skip-retrieval-judge", action="store_true")
    parser.add_argument("--rebuild-kb", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    cfg_pre = load_phase2_config(args.config)
    log_file = Path(cfg_pre["output"].get("pilot_log", "outputs/phase2/phase2_run.log"))
    _setup_logging(log_file)

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    structures = [s.strip() for s in args.structures.split(",")] if args.structures else None

    run_phase2(
        config_path=args.config,
        sample_size=args.sample_size,
        models=models,
        structures=structures,
        skip_eval=args.skip_eval,
        skip_llm_judge=args.skip_llm_judge,
        skip_retrieval_judge=args.skip_retrieval_judge,
        eval_only=args.eval_only,
        rebuild_kb=args.rebuild_kb,
        resume=not args.no_resume,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
