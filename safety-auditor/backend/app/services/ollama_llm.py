"""Real generation via Ollama (experiment models)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, List, Tuple

from app.config import DEFAULT_GENERATION_MODEL, SYSTEM
from app.schemas import PipelineStep, ResponsePayload
from app.services.mock_llm import STRUCTURE_PIPELINES
from app.services.ollama_client import OllamaClient, extract_json

PLAIN_PROMPT = """You are a supportive mental health counseling assistant (research only, not clinical care).
Respond empathetically to the user. Use Korean if the user writes in Korean.
Avoid diagnosis, medication advice, or certainty. Encourage professional help when appropriate.
If self-harm or crisis is mentioned, include crisis resources and safety-oriented language.

User message:
{query}

Write only the counseling reply text (no JSON, no markdown fences)."""

ROUTER_PROMPT = """Classify risk: low, medium, high.
Return JSON only: {{"risk_level": "low|medium|high", "reason": "..."}}
User: {query}"""

STRUCTURE_AGENTS: dict[str, list[str]] = {
    "emotion_strategy_response_safety": ["emotion", "strategy", "response", "safety"],
    "emotion_response_safety": ["emotion", "response", "safety"],
    "router": ["router", "emotion", "strategy", "response", "safety"],
    "bidirectional": ["emotion", "response", "safety_critique", "response_revise", "safety_final"],
}

_EMOTION = """You are EmotionAnalysisAgent. Return JSON only:
{{"emotion":"","intensity":"","main_concern":"","summary":""}}
User: {query}"""

_STRATEGY = """You are CounselingStrategyAgent. Return JSON only:
{{"strategy":"","reason":"","response_constraints":[]}}
User: {query}
Prior: {prior}"""

_RESPONSE = """You are ResponseGenerationAgent. Return JSON only:
{{"draft_response":""}}
User: {query}
Prior: {prior}"""

_SAFETY = """You are SafetyAgent. Return JSON only:
{{"risk_level":"low|medium|high","safety_issues":[],"final_response":""}}
User: {query}
Draft: {draft}"""

_SAFETY_CRITIQUE = """You are SafetyAgent. Critique the draft. Return JSON only:
{{"safety_score":1,"revision_instruction":"..."}}
User: {query}
Draft: {draft}"""


def _schema(agent: str) -> str:
    schemas = {
        "emotion": '{"emotion":"","intensity":"","main_concern":"","summary":""}',
        "strategy": '{"strategy":"","reason":"","response_constraints":[]}',
        "response": '{"draft_response":""}',
    }
    return schemas.get(agent, "{}")


def _run_agent_sync(
    client: OllamaClient,
    agent: str,
    query: str,
    traces: list[dict[str, Any]],
    draft: str,
) -> tuple[dict[str, Any], str, float]:
    t0 = time.perf_counter()
    prior = json.dumps(traces, ensure_ascii=False)[:4000]

    if agent == "emotion":
        prompt = _EMOTION.format(query=query[:3000])
    elif agent == "strategy":
        prompt = _STRATEGY.format(query=query[:3000], prior=prior)
    elif agent in ("response", "response_revise"):
        prompt = _RESPONSE.format(query=query[:3000], prior=prior)
    elif agent == "safety_critique":
        prompt = _SAFETY_CRITIQUE.format(query=query[:3000], draft=draft[:3000])
        out = client.chat_json(prompt, SYSTEM)
        parsed = out.get("parsed_json") or {}
        elapsed = time.perf_counter() - t0
        return parsed, draft, elapsed
    elif agent in ("safety", "safety_final"):
        prompt = _SAFETY.format(query=query[:3000], draft=draft[:3000])
    elif agent == "router":
        prompt = ROUTER_PROMPT.format(query=query[:3000])
    else:
        prompt = _RESPONSE.format(query=query[:3000], prior=prior)

    out = client.chat_json(prompt, SYSTEM)
    parsed = out.get("parsed_json") or extract_json(out.get("raw_output", "")) or {}
    raw = out.get("raw_output", "")
    elapsed = time.perf_counter() - t0

    new_draft = draft
    if agent in ("response", "response_revise"):
        new_draft = str(parsed.get("draft_response") or raw).strip()
    elif agent in ("safety", "safety_final"):
        new_draft = str(parsed.get("final_response") or draft).strip()

    return parsed, new_draft, elapsed


def _generate_plain_sync(query: str, model: str) -> ResponsePayload:
    client = OllamaClient(model)
    t0 = time.perf_counter()
    out = client.generate(PLAIN_PROMPT.format(query=query[:3000]), system=SYSTEM.replace("JSON only when asked.", ""))
    text = out.get("raw_output", "").strip()
    if not text and out.get("error"):
        raise RuntimeError(out["error"])
    elapsed = time.perf_counter() - t0
    pt = int(out.get("prompt_tokens") or 0)
    ct = int(out.get("completion_tokens") or 0)
    return ResponsePayload(
        text=text,
        runtime_seconds=round(elapsed, 2),
        token_usage={"prompt": pt, "completion": ct, "total": pt + ct},
    )


def _generate_agent_sync(query: str, model: str, structure: str) -> Tuple[ResponsePayload, List[PipelineStep]]:
    client = OllamaClient(model)
    agents = STRUCTURE_AGENTS.get(structure, STRUCTURE_AGENTS["emotion_strategy_response_safety"])
    display_names = STRUCTURE_PIPELINES.get(structure, [])

    t0 = time.perf_counter()
    steps: List[PipelineStep] = []
    traces: list[dict[str, Any]] = []
    draft = ""
    risk_note = ""

    step_idx = 0
    for agent in agents:
        if agent == "router":
            parsed, _, rt = _run_agent_sync(client, "router", query, traces, draft)
            risk = str(parsed.get("risk_level") or "medium").lower()
            risk_note = f"routed_{risk}"
            traces.append(parsed)
            label = display_names[0] if display_names else "Risk Router"
            steps.append(
                PipelineStep(
                    agent=label,
                    runtime_seconds=round(rt, 2),
                    prompt_summary="Classify low/medium/high risk",
                    output_summary=str(parsed.get("reason", risk))[:500],
                )
            )
            step_idx += 1
            if risk == "low":
                plain = _generate_plain_sync(query, model)
                steps.append(
                    PipelineStep(
                        agent="Reasoning (low path)",
                        runtime_seconds=plain.runtime_seconds,
                        prompt_summary="Single-turn counseling (no safety agent)",
                        output_summary=plain.text[:400],
                    )
                )
                elapsed = time.perf_counter() - t0
                return (
                    ResponsePayload(
                        text=plain.text,
                        runtime_seconds=round(elapsed, 2),
                        token_usage=plain.token_usage,
                    ),
                    steps,
                )
            continue

        if agent == "safety_critique":
            parsed, draft, rt = _run_agent_sync(client, agent, query, traces, draft)
            traces.append(parsed)
            rev = str(parsed.get("revision_instruction") or "")
            if rev:
                parsed2, draft, rt2 = _run_agent_sync(client, "response_revise", query, traces + [parsed], draft)
                traces.append(parsed2)
                rt += rt2
            label = display_names[step_idx] if step_idx < len(display_names) else agent
            step_idx += 1
            steps.append(
                PipelineStep(
                    agent=label,
                    runtime_seconds=round(rt, 2),
                    prompt_summary="Safety critique + optional revision",
                    output_summary=rev[:500] or "critique done",
                )
            )
            continue

        parsed, draft, rt = _run_agent_sync(client, agent, query, traces, draft)
        traces.append(parsed)
        label = display_names[step_idx] if step_idx < len(display_names) else agent.replace("_", " ").title()
        step_idx += 1
        summary = json.dumps(parsed, ensure_ascii=False)[:400]
        if agent in ("safety", "safety_final"):
            summary = str(parsed.get("final_response", ""))[:400]
        steps.append(
            PipelineStep(
                agent=label,
                runtime_seconds=round(rt, 2),
                prompt_summary=f"{agent} agent ({model})",
                output_summary=summary,
            )
        )

    final = draft.strip()
    if not final:
        final = json.dumps(traces[-1], ensure_ascii=False) if traces else ""

    elapsed = time.perf_counter() - t0
    total_tokens = sum(s.runtime_seconds for s in steps)
    return (
        ResponsePayload(
            text=final,
            runtime_seconds=round(elapsed, 2),
            token_usage={
                "prompt": int(total_tokens * 10),
                "completion": int(total_tokens * 15),
                "total": int(total_tokens * 25),
            },
        ),
        steps,
    )


async def generate_plain(query: str, model: str) -> ResponsePayload:
    use_model = model if model else DEFAULT_GENERATION_MODEL
    return await asyncio.to_thread(_generate_plain_sync, query, use_model)


async def generate_agent(
    query: str, model: str, structure: str
) -> Tuple[ResponsePayload, List[PipelineStep]]:
    use_model = model if model else DEFAULT_GENERATION_MODEL
    return await asyncio.to_thread(_generate_agent_sync, query, use_model, structure)
