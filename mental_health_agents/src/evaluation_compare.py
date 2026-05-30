"""Phase H: Compare base vs fine-tuned models on held-out cases."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data_loader import load_cases
from src.evaluation import compute_run_metrics
from src.experiments import run_case_with_config
from src.ollama_client import OllamaClient
from src.run_config import RunConfig
from src.training_paths import training_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _checkpoint_variants(cfg: dict[str, Any]) -> list[dict[str, str]]:
    paths = training_paths(cfg)
    ckpt_root = paths["checkpoints"]
    variants = [{"label": "Base Qwen", "model": cfg["ollama"]["model"], "type": "base"}]
    if not ckpt_root.exists():
        return variants

    for d in sorted(ckpt_root.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "training_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mode = meta.get("mode", "lora")
        label = f"{mode.upper()} ({meta.get('scale', '?')})"
        if meta.get("stub"):
            label += " [stub]"
        variants.append(
            {
                "label": label,
                "model": meta.get("model", cfg["ollama"]["model"]),
                "type": mode,
                "checkpoint": str(d),
            }
        )
    return variants


def evaluate_variant(
    label: str,
    model: str,
    cases: list[dict],
    structure: str = "single_llm",
) -> dict[str, float]:
    cfg = load_config()
    cfg = dict(cfg)
    cfg["ollama"] = {**cfg["ollama"], "model": model}
    client = OllamaClient(cfg)

    results = []
    for case in cases:
        run_cfg = RunConfig.from_structure(
            phase="eval_compare",
            structure=structure,
            model=model,
            sample_size=len(cases),
        )
        t0 = time.perf_counter()
        from src.database import Database
        from src.rag import KnowledgeRAG

        db = Database()
        rag = KnowledgeRAG(cfg) if run_cfg.use_rag else None
        row = run_case_with_config(run_cfg, case, client, rag, db)
        row["runtime_sec"] = time.perf_counter() - t0
        results.append(row)

    metrics = compute_run_metrics(results)
    metrics["variant"] = label
    metrics["model"] = model
    return metrics


def run_comparison(smoke: bool = False) -> pd.DataFrame:
    cfg = load_config()
    paths = training_paths(cfg)
    paths["reports"].mkdir(parents=True, exist_ok=True)

    n = cfg.get("training", {}).get("smoke_max_samples", 20) if smoke else 20
    cases = load_cases(sample_size=n, force_csv=True)

    rows = []
    for v in _checkpoint_variants(cfg):
        if v["type"] != "base" and v.get("checkpoint", "").endswith("_stub"):
            rows.append(
                {
                    "variant": v["label"],
                    "model": v["model"],
                    "symptom_f1": 0.0,
                    "risk_accuracy": 0.0,
                    "high_risk_recall": 0.0,
                    "unsafe_flag_rate": 0.0,
                    "hallucination_flag_rate": 0.0,
                    "json_parse_success_rate": 0.0,
                    "avg_runtime_sec": 0.0,
                    "note": "stub checkpoint — training not run",
                }
            )
            continue
        logger.info("Evaluating %s", v["label"])
        try:
            m = evaluate_variant(v["label"], v["model"], cases[: min(5, len(cases))] if smoke else cases)
            rows.append(m)
        except Exception as e:
            logger.warning("Eval failed for %s: %s", v["label"], e)
            rows.append({"variant": v["label"], "error": str(e)})

    df = pd.DataFrame(rows)
    csv_path = paths["reports"] / "model_comparison.csv"
    df.to_csv(csv_path, index=False)

    md_lines = [
        "# Training Evaluation Summary",
        "",
        "Comparison of base Ollama inference vs fine-tuned checkpoints (research evaluation only).",
        "",
        "## Model Comparison",
        "",
        df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(),
        "",
        "## Metrics",
        "- Symptom F1, Risk Accuracy, High-Risk Recall",
        "- Unsafe / Hallucination rates (lower is better)",
        "- JSON parse success, inference time",
        "",
    ]
    (paths["reports"] / "training_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Wrote %s and training_summary.md", csv_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Few cases, quick compare")
    args = parser.parse_args()
    run_comparison(smoke=args.smoke)


if __name__ == "__main__":
    main()
