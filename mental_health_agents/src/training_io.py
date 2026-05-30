"""JSONL I/O and train/valid/test splitting for training datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from src.utils import ensure_dir


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> int:
    path = Path(path)
    ensure_dir(str(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    for row in read_jsonl(path):
        yield row


def split_dataset(
    rows: list[dict[str, Any]],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not rows:
        return [], [], []
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    train = shuffled[:n_train]
    valid = shuffled[n_train : n_train + n_valid]
    test = shuffled[n_train + n_valid :]
    return train, valid, test


def write_splits(
    base_dir: Path,
    prefix: str,
    rows: list[dict[str, Any]],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, int]:
    train, valid, test = split_dataset(rows, ratios, seed)
    counts = {}
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        p = base_dir / f"{prefix}_{name}.jsonl"
        counts[name] = write_jsonl(p, part)
    return counts
