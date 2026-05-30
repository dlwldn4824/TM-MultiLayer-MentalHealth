#!/usr/bin/env python3
"""Sequential research workflow: inference artifacts → datasets → training → evaluation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset_builder import run_stage
from src.evaluation_compare import run_comparison
from src.training_runner import main as training_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STAGES = (
    "datasets",
    "lora_smoke",
    "dpo_smoke",
    "distill_smoke",
    "evaluate",
    "all_smoke",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Training expansion pipeline")
    parser.add_argument("--stage", choices=STAGES, default="all_smoke")
    parser.add_argument("--model", default="qwen2.5:7b")
    args = parser.parse_args()

    if args.stage in ("datasets", "all_smoke"):
        logger.info("Stage 2–3: Dataset generation (smoke)")
        run_stage("all", smoke=True)

    if args.stage in ("lora_smoke", "all_smoke"):
        logger.info("Stage 4: LoRA smoke training")
        sys.argv = ["training_runner", "--model", args.model, "--mode", "lora", "--scale", "smoke"]
        training_main()

    if args.stage in ("dpo_smoke", "all_smoke"):
        logger.info("Stage 5: DPO smoke training")
        sys.argv = ["training_runner", "--model", args.model, "--mode", "dpo", "--scale", "smoke"]
        training_main()

    if args.stage in ("distill_smoke", "all_smoke"):
        logger.info("Stage 6: Distillation smoke training")
        sys.argv = ["training_runner", "--model", args.model, "--mode", "distill", "--scale", "smoke"]
        training_main()

    if args.stage in ("evaluate", "all_smoke"):
        logger.info("Stage 7: Evaluation compare (smoke)")
        run_comparison(smoke=True)

    logger.info("Training pipeline stage '%s' finished.", args.stage)


if __name__ == "__main__":
    main()
