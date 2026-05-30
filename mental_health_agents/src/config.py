"""Load and expose project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _resolve_paths(cfg)


def _resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve relative paths against project root."""
    for key in ("sample_cases_path",):
        if key in cfg.get("data", {}):
            p = Path(cfg["data"][key])
            if not p.is_absolute():
                cfg["data"][key] = str(PROJECT_ROOT / p)

    kb = cfg.get("knowledge_base", {})
    for key in ("root", "sources_file"):
        if key in kb:
            p = Path(kb[key])
            if not p.is_absolute():
                kb[key] = str(PROJECT_ROOT / p)

    rag = cfg.get("rag", {})
    for key in ("chroma_dir",):
        if key in rag:
            p = Path(rag[key])
            if not p.is_absolute():
                rag[key] = str(PROJECT_ROOT / p)

    out = cfg.get("output", {})
    for key in (
        "sqlite_path",
        "predictions_csv",
        "metrics_json",
        "metrics_csv",
        "experiment_summary_md",
        "phase_state_json",
    ):
        if key in out:
            p = Path(out[key])
            if not p.is_absolute():
                out[key] = str(PROJECT_ROOT / p)
    tcfg = cfg.get("training", {})
    root = tcfg.get("root", "training")
    base = PROJECT_ROOT / root if not Path(root).is_absolute() else Path(root)
    for key in ("failure_dir", "sft_dir", "dpo_dir", "distillation_dir", "checkpoints_dir", "reports_dir"):
        if key in tcfg:
            p = Path(tcfg[key])
            if not p.is_absolute():
                tcfg[key] = str(base / p)
    return cfg


def get_config() -> dict[str, Any]:
    if not hasattr(get_config, "_cache"):
        get_config._cache = load_config()  # type: ignore[attr-defined]
    return get_config._cache  # type: ignore[attr-defined]
