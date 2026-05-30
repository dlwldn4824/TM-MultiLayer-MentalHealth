"""Resolved paths for training expansion artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, load_config


def training_root(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    root = Path(cfg.get("training", {}).get("root", "training"))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root


def training_paths(cfg: dict[str, Any] | None = None) -> dict[str, Path]:
    cfg = cfg or load_config()
    tcfg = cfg.get("training", {})
    root = training_root(cfg)

    def p(key: str, default: str) -> Path:
        rel = tcfg.get(key, default)
        path = Path(rel)
        return path if path.is_absolute() else root / path

    return {
        "root": root,
        "failure": p("failure_dir", "failure_dataset"),
        "sft": p("sft_dir", "sft_dataset"),
        "dpo": p("dpo_dir", "dpo_dataset"),
        "distillation": p("distillation_dir", "distillation_dataset"),
        "checkpoints": p("checkpoints_dir", "checkpoints"),
        "reports": p("reports_dir", "reports"),
    }
