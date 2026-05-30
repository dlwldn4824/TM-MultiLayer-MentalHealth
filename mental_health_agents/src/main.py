"""CLI entry point for running experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on path when run as module
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_cases
from src.evaluation import run_evaluation_pipeline
from src.experiments import run_all_experiments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mental health multi-agent experiment runner")
    p.add_argument("--sample-size", type=int, default=None, help="Override config sample_size")
    p.add_argument("--force-csv", action="store_true", help="Use sample_cases.csv only")
    p.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Subset of experiments to run",
    )
    p.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(
        sample_size=args.sample_size,
        config_path=args.config,
        force_csv=args.force_csv,
    )
    logger.info("Loaded %d cases", len(cases))
    results = run_all_experiments(
        cases,
        experiment_names=args.experiments,
        config_path=args.config,
    )
    metrics = run_evaluation_pipeline(results)
    logger.info("Evaluation complete. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
