#!/usr/bin/env python3
"""Run Sequential Experimental Design phases (0–6) with pruning."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.phases import PhasePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential phase experiment pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--only-phase",
        type=str,
        default=None,
        choices=[
            "phase0_smoke_test",
            "phase1_structure_comparison",
            "phase2_model_comparison",
            "phase3_ensemble_comparison",
            "phase4_weight_search",
            "phase5_ablation",
            "phase6_data_scaling",
        ],
        help="Run a single phase only",
    )
    parser.add_argument(
        "--disable-phases",
        action="store_true",
        help="Use experiment_phases flags in config only (default)",
    )
    args = parser.parse_args()

    pipeline = PhasePipeline(config_path=args.config, only_phase=args.only_phase)
    state = pipeline.run()
    print("Pipeline complete. Phase state saved.")
    print(f"phase0_passed: {state.get('phase0_passed')}")
    print(f"phase1_best_structure: {state.get('phase1_best_structure')}")
    print(f"phase2_best_model: {state.get('phase2_best_model')}")


if __name__ == "__main__":
    main()
