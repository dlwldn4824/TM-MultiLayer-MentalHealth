"""Load CounselChat from Hugging Face (no pseudo-labels)."""

from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset


def load_counsel_cases(
    dataset_id: str = "nbertagnolli/counsel-chat",
    split: str = "train",
    sample_size: int = 30,
    seed: int = 42,
) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_id, split=split)
    valid: list[dict[str, Any]] = []
    for row in ds:
        q = (row.get("questionText") or "").strip()
        ref = (row.get("answerText") or "").strip()
        if len(q) < 20 or len(ref) < 40:
            continue
        valid.append(row)

    rng = random.Random(seed)
    rng.shuffle(valid)
    selected = valid[:sample_size]

    cases: list[dict[str, Any]] = []
    for i, row in enumerate(selected):
        cases.append(
            {
                "case_id": str(row.get("questionID", i)),
                "question_text": (row.get("questionText") or "").strip(),
                "reference_answer": (row.get("answerText") or "").strip(),
                "topic": row.get("topic") or "",
                "question_title": row.get("questionTitle") or "",
            }
        )
    return cases
