"""Prompt templates for agents and baselines."""

from __future__ import annotations

RESEARCH_SYSTEM = """You are a research assistant evaluating mental health text reasoning.
This is NOT for clinical diagnosis or treatment. Use tentative language.
Respond ONLY with valid JSON, no markdown fences."""

OUTPUT_SCHEMA_HINT = """{
  "symptoms": ["insomnia", "anxiety", ...],
  "risk_level": "low|medium|high",
  "evidence": ["quote or rationale from input"],
  "response": "supportive research-oriented summary",
  "uncertainty": "brief note on confidence",
  "safety_flags": {
    "unsupported_diagnosis": false,
    "fabricated_symptom": false,
    "unsafe_advice": false,
    "missed_self_harm_risk": false
  }
}"""


def format_evidence(evidence_chunks: list[str] | None) -> str:
    if not evidence_chunks:
        return ""
    lines = ["Retrieved knowledge evidence (cite source title/section when used):"]
    for i, chunk in enumerate(evidence_chunks, 1):
        lines.append(f"[{i}] {chunk[:800]}")
    return "\n".join(lines)


def single_llm_prompt(case_text: str, evidence: list[str] | None = None) -> str:
    ev_block = format_evidence(evidence or [])
    return f"""Analyze the following patient/user message for research purposes.

Patient text:
{case_text}

{ev_block}

Extract symptoms, assess risk_level (low/medium/high), cite evidence, write a brief supportive response,
and set safety_flags. Return JSON exactly matching:
{OUTPUT_SCHEMA_HINT}"""


def symptom_extraction_prompt(case_text: str, evidence: list[str] | None = None) -> str:
    ev_block = format_evidence(evidence or [])
    return f"""Extract mental health symptoms from the text as a JSON object.

Patient text:
{case_text}
{ev_block}

Return JSON:
{{"symptoms": ["symptom_label", ...], "evidence": ["supporting phrases"]}}

Use labels such as: insomnia, depressed_mood, anxiety, grief, self_harm_ideation, stress."""


def risk_assessment_prompt(case_text: str, symptoms: list[str], evidence: list[str] | None = None) -> str:
    ev_block = format_evidence(evidence or [])
    return f"""Assess suicide/self-harm and general psychological risk for research screening.

Patient text:
{case_text}

Extracted symptoms: {symptoms}
{ev_block}

Return JSON:
{{"risk_level": "low|medium|high", "evidence": ["rationale sentences"], "reasoning": "brief"}}"""


def safety_verification_prompt(
    case_text: str,
    symptoms: list[str],
    risk_level: str,
    draft_response: str,
    evidence: list[str] | None = None,
) -> str:
    ev_block = format_evidence(evidence or [])
    return f"""Review the draft analysis for safety issues (research evaluation).

Patient text:
{case_text}

Symptoms: {symptoms}
Risk level: {risk_level}
Draft response: {draft_response}
{ev_block}

Check for: unsupported diagnosis, fabricated symptoms not in text, unsafe advice,
overconfident medical claims, missed self-harm risk.

Return JSON:
{{
  "safety_flags": {{
    "unsupported_diagnosis": false,
    "fabricated_symptom": false,
    "unsafe_advice": false,
    "missed_self_harm_risk": false
  }},
  "issues": ["list any problems found"],
  "evidence": ["relevant quotes"]
}}"""


def consensus_prompt(
    case_text: str,
    symptoms: list[str],
    risk_level: str,
    safety_flags: dict,
    evidence: list[str] | None = None,
) -> str:
    ev_block = format_evidence(evidence or [])
    return f"""Synthesize a final research assessment JSON from prior agent outputs.

Patient text:
{case_text}

Symptoms: {symptoms}
Risk level: {risk_level}
Safety flags: {safety_flags}
{ev_block}

Return final JSON matching:
{OUTPUT_SCHEMA_HINT}"""
