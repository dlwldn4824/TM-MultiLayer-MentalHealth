"""Load MentalChat16K or fallback CSV cases."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.pseudo_labeler import apply_pseudo_labels

logger = logging.getLogger(__name__)


def _extract_text_from_row(row: dict[str, Any]) -> str:
    for key in ("input", "user", "patient", "question", "text", "instruction"):
        val = row.get(key)
        if val and isinstance(val, str) and len(val.strip()) > 20:
            if key == "instruction" and row.get("input"):
                continue
            return val.strip()
    parts = [str(v) for v in row.values() if isinstance(v, str) and len(v) > 20]
    return parts[0] if parts else ""


def load_from_huggingface(dataset_name: str, sample_size: int, seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name)
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    data = ds[split_name]

    rows = []
    for i, row in enumerate(data):
        text = _extract_text_from_row(dict(row))
        if not text or len(text) < 10:
            continue
        rows.append(
            {
                "case_id": f"mc16k_{i}",
                "text": text,
                "source": dataset_name,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid rows extracted from HuggingFace dataset")

    df = df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True)
    df["case_id"] = [f"mc16k_{i}" for i in range(len(df))]
    return df.to_dict(orient="records")


def load_from_csv(csv_path: str, sample_size: int, seed: int) -> list[dict[str, Any]]:
    path = Path(csv_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError(f"CSV must have 'text' column: {path}")
    if "case_id" not in df.columns:
        df["case_id"] = [f"sample_{i}" for i in range(len(df))]
    if "source" not in df.columns:
        df["source"] = "sample_cases"
    df = df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True)
    return df.to_dict(orient="records")


def load_cases(
    sample_size: int | None = None,
    config_path: str | None = None,
    force_csv: bool = False,
) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    n = sample_size or data_cfg["sample_size"]
    seed = data_cfg.get("random_seed", 42)
    cases: list[dict[str, Any]] = []

    if not force_csv:
        try:
            cases = load_from_huggingface(data_cfg["dataset_name"], n, seed)
            logger.info("Loaded %d cases from %s", len(cases), data_cfg["dataset_name"])
        except Exception as e:
            logger.warning("HuggingFace load failed (%s), using CSV fallback", e)

    if not cases:
        csv_path = data_cfg["sample_cases_path"]
        cases = load_from_csv(csv_path, n, seed)
        logger.info("Loaded %d cases from %s", len(cases), csv_path)

    return [apply_pseudo_labels(c) for c in cases]
