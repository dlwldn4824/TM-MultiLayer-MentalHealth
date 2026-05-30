"""Weak / pseudo labeling from case text."""

from __future__ import annotations

import re
from typing import Any

HIGH_RISK_PATTERNS = [
    r"\bsuicide\b",
    r"\bkill myself\b",
    r"\bself[- ]?harm\b",
    r"\bhurt myself\b",
    r"\bend my life\b",
    r"\bwant to die\b",
    r"\btake my life\b",
]

MEDIUM_RISK_PATTERNS = [
    r"\bhopeless\b",
    r"\bcan'?t go on\b",
    r"\bno reason to live\b",
    r"\bnothing matters\b",
    r"\bdon'?t see any point\b",
]

LOW_MEDIUM_SYMPTOM_PATTERNS = {
    "anxiety": [r"\banxious\b", r"\bpanic\b", r"\bworr(y|ied|ying)\b", r"\bnervous\b"],
    "depressed_mood": [r"\bdepressed\b", r"\bsad\b", r"\bhopeless\b", r"\blow mood\b"],
    "insomnia": [r"\bcan'?t sleep\b", r"\bsleepless\b", r"\binsomnia\b", r"\blie awake\b", r"\btrouble sleeping\b", r"\bsleep\b"],
    "grief": [r"\bloss\b", r"\bpassed away\b", r"\bdied\b", r"\bbereavement\b", r"\bgrief\b", r"\bmourning\b"],
    "self_harm_ideation": HIGH_RISK_PATTERNS,
    "stress": [r"\bstressed\b", r"\boverwhelmed\b", r"\bburnout\b", r"\bpressure\b"],
}


def _match_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def label_risk(text: str) -> str:
    t = text.lower()
    if _match_any(t, HIGH_RISK_PATTERNS):
        return "high"
    if _match_any(t, MEDIUM_RISK_PATTERNS):
        return "medium"
    for label, patterns in LOW_MEDIUM_SYMPTOM_PATTERNS.items():
        if label == "self_harm_ideation":
            continue
        if _match_any(t, patterns):
            if label in ("depressed_mood", "grief") and _match_any(t, MEDIUM_RISK_PATTERNS):
                return "medium"
            return "low"
    return "low"


def label_symptoms(text: str) -> list[str]:
    t = text.lower()
    found = []
    for symptom, patterns in LOW_MEDIUM_SYMPTOM_PATTERNS.items():
        if _match_any(t, patterns):
            found.append(symptom)
    return sorted(set(found))


def apply_pseudo_labels(case: dict[str, Any]) -> dict[str, Any]:
    text = case.get("text", "")
    case["risk_label"] = label_risk(text)
    case["symptom_labels"] = label_symptoms(text)
    return case
