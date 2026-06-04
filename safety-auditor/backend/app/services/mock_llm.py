"""Mock LLM responses and agent pipelines (no API keys required)."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from app.schemas import PipelineStep, ResponsePayload

STRUCTURE_PIPELINES: dict[str, list[str]] = {
    "emotion_strategy_response_safety": [
        "Emotion Analysis",
        "Strategy Planning",
        "Response Generation",
        "Safety Check",
    ],
    "emotion_response_safety": [
        "Emotion Analysis",
        "Response Generation",
        "Safety Check",
    ],
    "router": [
        "Risk Router",
        "Routed Pipeline",
        "Safety Check",
    ],
    "bidirectional": [
        "Emotion Analysis",
        "Response Draft",
        "Safety Critique",
        "Response Revision",
        "Safety Final",
    ],
}

PLAIN_TEMPLATES = [
    "불안하고 무기력한 감정은 흔히 겪을 수 있는 반응입니다. 자신을 탓하기보다, "
    "충분한 휴식과 가까운 사람과의 대화를 시도해 보시길 권합니다. "
    "증상이 지속되면 정신건강 전문가 상담을 고려해 주세요.",
    "말씀해 주신 감정은 충분히 이해됩니다. 일상에서 작은 목표를 하나씩 설정하고, "
    "규칙적인 수면·식사를 유지하는 것이 도움이 될 수 있습니다. "
    "전문 상담이 필요하다고 느끼시면 주저하지 마시고 도움을 요청하세요.",
]

AGENT_TEMPLATES = [
    "지금 느끼시는 불안과 무기력함은 많은 분이 겪는 어려움입니다. "
    "‘이상한 것’이 아니라, 지금 상황에서 자연스러운 반응일 수 있어요. "
    "짧은 산책이나 호흡 연습처럼 부담 없는 활동부터 시작해 보시고, "
    "혼자 감당하기 어렵다면 신뢰할 수 있는 상담 전문가와 상의하시길 권해 드립니다. "
    "자해·극단적 생각이 드시면 즉시 위기 상담 연락처(예: 1393)로 연락해 주세요.",
    "감정을 표현해 주셔서 감사합니다. 불안과 무기력은 회복 과정에서 흔히 나타납니다. "
    "자기 비난보다 자기 돌봄에 초점을 맞추시고, 증상이 악화되거나 위기감이 있으면 "
    "반드시 전문가의 도움을 받으시길 바랍니다.",
]


async def _delay(base: float = 0.4) -> float:
    t = base + random.uniform(0.1, 0.5)
    await asyncio.sleep(t)
    return t


async def generate_plain(query: str, model: str) -> ResponsePayload:
    t0 = time.perf_counter()
    await _delay(0.8)
    text = PLAIN_TEMPLATES[hash(query) % len(PLAIN_TEMPLATES)]
    if "불안" in query or "anxious" in query.lower():
        text = (
            f"[{model} · Single] 말씀하신 불안과 무기력함은 이해할 수 있는 반응입니다. "
            "자신을 이상하다고 느끼지 않으셔도 됩니다. 규칙적인 생활 리듬과 "
            "가벼운 활동을 시도해 보시고, 증상이 지속되면 전문 상담을 권합니다."
        )
    elapsed = time.perf_counter() - t0
    tokens = 120 + len(query) // 2
    return ResponsePayload(
        text=text,
        runtime_seconds=round(elapsed, 2),
        token_usage={"prompt": tokens // 3, "completion": tokens * 2 // 3, "total": tokens},
    )


async def generate_agent(
    query: str,
    model: str,
    structure: str,
) -> tuple[ResponsePayload, list[PipelineStep]]:
    t0 = time.perf_counter()
    steps_def = STRUCTURE_PIPELINES.get(
        structure, STRUCTURE_PIPELINES["emotion_strategy_response_safety"]
    )
    steps: list[PipelineStep] = []
    for agent in steps_def:
        rt = await _delay(0.35)
        steps.append(
            PipelineStep(
                agent=agent,
                runtime_seconds=round(rt, 2),
                prompt_summary=f"[{model}] {agent}: analyze user message ({len(query)} chars)",
                output_summary=f"{agent} completed with safety-aware counseling constraints.",
            )
        )
    text = AGENT_TEMPLATES[hash(query + structure) % len(AGENT_TEMPLATES)]
    if structure == "router":
        text = (
            f"[Router · {model}] 위험도 medium으로 분류 후 Emotion→Strategy→Response→Safety 경로를 적용했습니다. "
            + text
        )
    elif structure == "bidirectional":
        text = (
            f"[Bidirectional · {model}] Safety critique 반영 후 수정된 최종 응답: " + text
        )
    else:
        text = f"[Agent · {structure}] " + text

    elapsed = time.perf_counter() - t0
    tokens = 80 * len(steps_def) + len(query)
    return (
        ResponsePayload(
            text=text,
            runtime_seconds=round(elapsed, 2),
            token_usage={
                "prompt": tokens // 2,
                "completion": tokens // 2,
                "total": tokens,
            },
        ),
        steps,
    )
