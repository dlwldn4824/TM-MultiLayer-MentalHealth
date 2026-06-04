from __future__ import annotations

from app.schemas import InsightRequest, InsightResponse, JudgeScores


def _delta(a: float, b: float) -> float:
    return round(b - a, 1)


def generate_insights(req: InsightRequest) -> InsightResponse:
    p, ag = req.plain_scores, req.agent_scores
    bullets = [
        f"Safety {_delta(p.safety, ag.safety):+.1f}",
        f"Empathy {_delta(p.empathy, ag.empathy):+.1f}",
        f"Trust {_delta(p.trust, ag.trust):+.1f}",
        f"Faithfulness {_delta(p.faithfulness, ag.faithfulness):+.1f}",
        f"Runtime {_delta(req.runtime_plain, req.runtime_agent):+.1f}초",
    ]
    improved = sum(
        1
        for k in ("safety", "empathy", "trust", "helpfulness", "faithfulness", "coherence")
        if getattr(ag, k) > getattr(p, k)
    )
    summary = (
        f"Agent Structure({req.structure}) 적용 후 "
        f"안전성·신뢰도·충실도가 Single LLM 대비 개선된 항목이 {improved}개입니다. "
        f"Safety {_delta(p.safety, ag.safety):+.1f}, Trust {_delta(p.trust, ag.trust):+.1f}, "
        f"Faithfulness {_delta(p.faithfulness, ag.faithfulness):+.1f}. "
        f"다만 에이전트 파이프라인으로 인해 응답 시간이 "
        f"{_delta(req.runtime_plain, req.runtime_agent):+.1f}초 증가했습니다."
    )
    return InsightResponse(summary=summary, bullets=bullets)
