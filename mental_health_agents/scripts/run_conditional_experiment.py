#!/usr/bin/env python3
"""Run Conditional Bidirectional on the same CounselChat cases as Phase 2."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ollama_client import OllamaClient
from src.pilot.counsel_data import load_counsel_cases
from src.pilot.evaluation import JUDGE_CRITERIA, evaluate_all
from src.pilot.kb import PilotRAG
from src.pilot.runners import (
    STRUCTURE_CONDITIONAL_BIDIRECTIONAL,
    STRUCTURE_SINGLE_RAG,
    STRUCTURE_THREE_AGENT,
    run_structure,
)

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
    "llm_calls",
    "revision_triggered",
    "gatekeeper_pass",
    "retrieval_json",
    "agent_trace_json",
]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: Path) -> set[tuple[str, str, str]]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as f:
        return {(r["case_id"], r["model"], r["structure"]) for r in csv.DictReader(f)}


def append_checkpoint(path: Path, row: dict) -> None:
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CHECKPOINT_COLUMNS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def summarize(
    rows: list[dict],
    baseline_rows: list[dict],
    out_dir: Path,
) -> None:
    cond = [r for r in rows if r["structure"] == STRUCTURE_CONDITIONAL_BIDIRECTIONAL]
    if not cond:
        return

    n = len(cond)
    rev_rate = 100.0 * sum(1 for r in cond if str(r.get("revision_triggered")).lower() == "true") / n
    pass_rate = 100.0 * sum(1 for r in cond if str(r.get("gatekeeper_pass")).lower() == "true") / n
    avg_calls = sum(int(r.get("llm_calls") or 0) for r in cond) / n
    avg_rt = sum(float(r.get("runtime_sec") or 0) for r in cond) / n

    base = {
        STRUCTURE_SINGLE_RAG: [r for r in baseline_rows if r["structure"] == STRUCTURE_SINGLE_RAG],
        STRUCTURE_THREE_AGENT: [r for r in baseline_rows if r["structure"] == STRUCTURE_THREE_AGENT],
    }
    lines = [
        "# Conditional Bidirectional Experiment",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Pipeline",
        "",
        "```",
        "Question → Retrieval → Response → Safety Gatekeeper",
        "  → (if issues) Revision → Safety Recheck → Final Answer",
        "  → (if pass) Final Answer = Response",
        "```",
        "",
        "## vs Phase 2 prototype (`three_agent`)",
        "",
        "| | three_agent (prototype) | conditional_bidirectional |",
        "|--|-------------------------|---------------------------|",
        "| Retrieval | RetrieverAgent + RAG | RAG only |",
        "| Response | ReasoningAgent | single_rag-style Response |",
        "| Safety | SafetyAgent (always revise-capable) | Gatekeeper → conditional Revision → Recheck |",
        "| LLM calls (pass path) | 3 | 3 |",
        "| LLM calls (revision path) | 3 | 5 |",
        "",
        "## Run stats (conditional_bidirectional)",
        "",
        f"- Cases: **{n}**",
        f"- Gatekeeper pass (no revision): **{pass_rate:.1f}%**",
        f"- Revision triggered: **{rev_rate:.1f}%**",
        f"- Avg LLM calls: **{avg_calls:.2f}**",
        f"- Avg runtime: **{avg_rt:.1f}s**",
        "",
    ]

    if base[STRUCTURE_THREE_AGENT]:
        t_rt = sum(float(r["runtime_sec"]) for r in base[STRUCTURE_THREE_AGENT]) / len(
            base[STRUCTURE_THREE_AGENT]
        )
        lines.append(f"- three_agent avg runtime (Phase 2): **{t_rt:.1f}s**")
    if base[STRUCTURE_SINGLE_RAG]:
        s_rt = sum(float(r["runtime_sec"]) for r in base[STRUCTURE_SINGLE_RAG]) / len(
            base[STRUCTURE_SINGLE_RAG]
        )
        lines.append(f"- single_rag avg runtime (Phase 2): **{s_rt:.1f}s**")

    lines.extend(["", "## Judge comparison (this run)", ""])
    for struct in (STRUCTURE_SINGLE_RAG, STRUCTURE_THREE_AGENT, STRUCTURE_CONDITIONAL_BIDIRECTIONAL):
        subset = [r for r in rows if r["structure"] == struct and r.get("_judge")]
        if not subset:
            continue
        lines.append(f"### {struct}")
        lines.append("")
        for k in JUDGE_CRITERIA:
            avg = sum(r["_judge"][k] for r in subset) / len(subset)
            lines.append(f"- {k}: **{avg:.2f}**")
        lines.append("")

    (out_dir / "conditional_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conditional_bidirectional experiment")
    parser.add_argument("--config", default="config_phase2.yaml")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--structures", default=STRUCTURE_CONDITIONAL_BIDIRECTIONAL)
    parser.add_argument("--sample-size", type=int, default=None, help="Override case count")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--output-dir", default="outputs/phase2/conditional_experiment")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg_path = PROJECT_ROOT / args.config
    cfg = load_config(cfg_path)
    cfg["ollama"]["model"] = args.model

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.csv"
    judge_path = out_dir / "judge_scores.csv"
    p2_pred = PROJECT_ROOT / "outputs/phase2/predictions.csv"

    structures = [s.strip() for s in args.structures.split(",") if s.strip()]
    p2 = cfg.get("phase2", {})
    cases = load_counsel_cases(
        dataset_id=p2.get("dataset_id", "nbertagnolli/counsel-chat"),
        split=p2.get("dataset_split", "train"),
        sample_size=int(args.sample_size or p2.get("sample_size", 30)),
        seed=int(p2.get("random_seed", 42)),
    )

    done = load_checkpoint(pred_path)
    rows: list[dict] = []
    if pred_path.is_file():
        rows = list(csv.DictReader(pred_path.open(encoding="utf-8")))

    if not args.eval_only:
        rag = PilotRAG(cfg)
        client = OllamaClient(cfg)
        todo = [
            (case, struct)
            for case in cases
            for struct in structures
            if (case["case_id"], args.model, struct) not in done
        ]
        logger.info("Inference: %d case×structure pairs to run", len(todo))
        for case, struct in tqdm(todo, desc="conditional"):
            result = run_structure(struct, client, rag, case)
            row = {
                "case_id": case["case_id"],
                "model": args.model,
                "structure": struct,
                "topic": case.get("topic", ""),
                "question_text": case["question_text"],
                "reference_answer": case["reference_answer"],
                "model_response": result["response"],
                "runtime_sec": f"{result.get('runtime_sec', 0):.2f}",
                "llm_calls": result.get("llm_calls", ""),
                "revision_triggered": result.get("revision_triggered", ""),
                "gatekeeper_pass": result.get("gatekeeper_pass", ""),
                "retrieval_json": json.dumps(result.get("retrieval") or {}, ensure_ascii=False),
                "agent_trace_json": json.dumps(result.get("agent_trace", []), ensure_ascii=False),
            }
            append_checkpoint(pred_path, row)
            rows.append(row)

    if not rows and pred_path.is_file():
        rows = list(csv.DictReader(pred_path.open(encoding="utf-8")))

    eval_rows = []
    for r in rows:
        eval_rows.append(
            {
                "case_id": r["case_id"],
                "structure": r["structure"],
                "question_text": r["question_text"],
                "reference_answer": r["reference_answer"],
                "model_response": r["model_response"],
                "runtime_sec": float(r.get("runtime_sec") or 0),
                "retrieval": json.loads(r.get("retrieval_json") or "{}"),
            }
        )

    client = OllamaClient(cfg)
    _, judge_df, _ = evaluate_all(eval_rows, client, cfg)
    judge_df["model"] = args.model
    judge_df.to_csv(judge_path, index=False)

    judge_by = {
        (str(r["case_id"]), r["structure"]): {k: float(r[k]) for k in JUDGE_CRITERIA}
        for _, r in judge_df.iterrows()
    }
    for r in rows:
        r["_judge"] = judge_by.get((str(r["case_id"]), r["structure"]), {})

    baseline_rows = []
    if p2_pred.is_file():
        baseline_rows = [
            r for r in csv.DictReader(p2_pred.open(encoding="utf-8")) if r["model"] == args.model
        ]

    summarize(rows, baseline_rows, out_dir)
    logger.info("Done. Outputs: %s", out_dir)


if __name__ == "__main__":
    main()
