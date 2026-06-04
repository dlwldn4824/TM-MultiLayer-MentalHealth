"""Mock judge evaluation (1-5 scores)."""

from __future__ import annotations

import hashlib
import random

from app.schemas import JudgeScores


def _seed(*parts: str) -> int:
    h = hashlib.md5("".join(parts).encode()).hexdigest()
    return int(h[:8], 16)


def judge_response(
    query: str,
    response: str,
    *,
    is_agent: bool,
    provider: str = "gpt-4o",
) -> JudgeScores:
    rng = random.Random(_seed(query, response, str(is_agent), provider))
    base = 3.8 if is_agent else 3.5
    def s() -> float:
        return round(min(5.0, max(1.0, base + rng.uniform(-0.4, 0.6))), 1)

    scores = JudgeScores(
        safety=s(),
        empathy=s(),
        trust=s(),
        helpfulness=s(),
        faithfulness=s(),
        coherence=s(),
    )
    if is_agent:
        scores.safety = min(5.0, scores.safety + 0.3)
        scores.trust = min(5.0, scores.trust + 0.2)
        scores.faithfulness = min(5.0, scores.faithfulness + 0.15)
    return scores
