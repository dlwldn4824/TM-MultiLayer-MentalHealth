"""Orchestrate training dataset generation (Phases A–G data stages)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.distillation_builder import build_distillation_dataset
from src.dpo_builder import build_dpo_dataset
from src.failure_analysis import build_failure_dataset
from src.sft_builder import build_sft_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STAGES = ("failure", "sft", "dpo", "distill", "all")


def run_stage(stage: str, smoke: bool = False) -> dict[str, int]:
    cfg = load_config()
    limit = cfg.get("training", {}).get("smoke_max_samples", 20) if smoke else None
    counts: dict[str, int] = {}

    if stage in ("failure", "all"):
        counts["failure_cases"] = build_failure_dataset()
    if stage in ("sft", "all"):
        counts.update(build_sft_datasets(smoke_limit=limit))
    if stage in ("dpo", "all"):
        counts["dpo_pairs"] = build_dpo_dataset(smoke_limit=limit)
    if stage in ("distill", "all"):
        counts.update(
            build_distillation_dataset(
                smoke_limit=limit,
                run_teacher_if_missing=False,
            )
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training datasets from experiment logs")
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="Which dataset stage to build",
    )
    parser.add_argument("--smoke", action="store_true", help="Limit samples for smoke test")
    args = parser.parse_args()
    counts = run_stage(args.stage, smoke=args.smoke)
    logger.info("Dataset build complete: %s", counts)


if __name__ == "__main__":
    main()
