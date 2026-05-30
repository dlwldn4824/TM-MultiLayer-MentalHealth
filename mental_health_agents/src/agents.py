"""Multi-agent pipeline and single-LLM baseline runners."""

from __future__ import annotations

import logging
from typing import Any

from src.ollama_client import OllamaClient
from src.prompts import (
    RESEARCH_SYSTEM,
    consensus_prompt,
    risk_assessment_prompt,
    safety_verification_prompt,
    single_llm_prompt,
    symptom_extraction_prompt,
)
from src.utils import merge_output, normalize_risk, normalize_symptoms

logger = logging.getLogger(__name__)


class SymptomExtractionAgent:
    name = "symptom_extraction"

    def __init__(self, client: OllamaClient):
        self.client = client

    def run(self, case_text: str, evidence: list[str] | None = None) -> dict[str, Any]:
        prompt = symptom_extraction_prompt(case_text, evidence)
        result = self.client.chat_json(prompt, RESEARCH_SYSTEM)
        parsed = result.get("parsed_json") or {}
        return {
            "symptoms": normalize_symptoms(parsed.get("symptoms")),
            "evidence": parsed.get("evidence", []) if isinstance(parsed.get("evidence"), list) else [],
            "raw_output": result.get("raw_output", ""),
            "parsed": parsed,
        }


class RiskAssessmentAgent:
    name = "risk_assessment"

    def __init__(self, client: OllamaClient):
        self.client = client

    def run(
        self, case_text: str, symptoms: list[str], evidence: list[str] | None = None
    ) -> dict[str, Any]:
        prompt = risk_assessment_prompt(case_text, symptoms, evidence)
        result = self.client.chat_json(prompt, RESEARCH_SYSTEM)
        parsed = result.get("parsed_json") or {}
        return {
            "risk_level": normalize_risk(parsed.get("risk_level")),
            "evidence": parsed.get("evidence", []) if isinstance(parsed.get("evidence"), list) else [],
            "raw_output": result.get("raw_output", ""),
            "parsed": parsed,
        }


class SafetyVerificationAgent:
    name = "safety_verification"

    def __init__(self, client: OllamaClient):
        self.client = client

    def run(
        self,
        case_text: str,
        symptoms: list[str],
        risk_level: str,
        draft_response: str,
        evidence: list[str] | None = None,
    ) -> dict[str, Any]:
        prompt = safety_verification_prompt(case_text, symptoms, risk_level, draft_response, evidence)
        result = self.client.chat_json(prompt, RESEARCH_SYSTEM)
        parsed = result.get("parsed_json") or {}
        flags = parsed.get("safety_flags", {})
        default_flags = {
            "unsupported_diagnosis": False,
            "fabricated_symptom": False,
            "unsafe_advice": False,
            "missed_self_harm_risk": False,
        }
        if isinstance(flags, dict):
            default_flags.update({k: bool(flags.get(k, False)) for k in default_flags})
        return {
            "safety_flags": default_flags,
            "issues": parsed.get("issues", []),
            "evidence": parsed.get("evidence", []) if isinstance(parsed.get("evidence"), list) else [],
            "raw_output": result.get("raw_output", ""),
            "parsed": parsed,
        }


class ConsensusAgent:
    name = "consensus"

    def __init__(self, client: OllamaClient):
        self.client = client

    def run(
        self,
        case_text: str,
        symptoms: list[str],
        risk_level: str,
        safety_flags: dict,
        evidence: list[str] | None = None,
    ) -> dict[str, Any]:
        prompt = consensus_prompt(case_text, symptoms, risk_level, safety_flags, evidence)
        result = self.client.chat_json(prompt, RESEARCH_SYSTEM)
        parsed = result.get("parsed_json")
        output = merge_output(parsed, result.get("raw_output", ""))
        output["safety_flags"] = {**output["safety_flags"], **safety_flags}
        return {
            "output": output,
            "raw_output": result.get("raw_output", ""),
            "parsed": parsed,
        }


def run_single_llm(
    client: OllamaClient,
    case_text: str,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    prompt = single_llm_prompt(case_text, evidence)
    result = client.chat_json(prompt, RESEARCH_SYSTEM)
    output = merge_output(result.get("parsed_json"), result.get("raw_output", ""))
    if evidence:
        output["evidence"] = list(output.get("evidence", [])) + evidence[:3]
    return {
        "output": output,
        "raw_output": result.get("raw_output", ""),
        "parsed_json": result.get("parsed_json"),
        "agent_name": "single_llm",
    }


def run_multi_agent(
    client: OllamaClient,
    case_text: str,
    evidence: list[str] | None = None,
    *,
    enable_symptom: bool = True,
    enable_risk: bool = True,
    enable_safety: bool = True,
    enable_consensus: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Symptom → Risk → Safety → Consensus. Ablation via enable_* flags."""
    traces: list[dict[str, Any]] = []

    if enable_symptom:
        symptom_agent = SymptomExtractionAgent(client)
        s_out = symptom_agent.run(case_text, evidence)
        traces.append({"agent": symptom_agent.name, **s_out})
        symptoms = s_out["symptoms"]
    else:
        s_out = {"symptoms": [], "evidence": [], "raw_output": "", "parsed": {}}
        traces.append({"agent": "symptom_extraction_skipped", **s_out})
        symptoms = []

    if enable_risk:
        risk_agent = RiskAssessmentAgent(client)
        r_out = risk_agent.run(case_text, symptoms, evidence)
        traces.append({"agent": risk_agent.name, **r_out})
        risk_level = r_out["risk_level"]
    else:
        r_out = {"risk_level": "medium", "evidence": [], "raw_output": "", "parsed": {}}
        traces.append({"agent": "risk_assessment_skipped", **r_out})
        risk_level = "medium"

    default_flags = {
        "unsupported_diagnosis": False,
        "fabricated_symptom": False,
        "unsafe_advice": False,
        "missed_self_harm_risk": False,
    }

    if enable_safety:
        draft = f"Symptoms: {symptoms}. Risk: {risk_level}."
        safety_agent = SafetyVerificationAgent(client)
        sf_out = safety_agent.run(case_text, symptoms, risk_level, draft, evidence)
        traces.append({"agent": safety_agent.name, **sf_out})
        safety_flags = sf_out["safety_flags"]
    else:
        sf_out = {"safety_flags": default_flags, "raw_output": "", "parsed": {}}
        traces.append({"agent": "safety_verification_skipped", **sf_out})
        safety_flags = default_flags

    if enable_consensus:
        consensus_agent = ConsensusAgent(client)
        c_out = consensus_agent.run(case_text, symptoms, risk_level, safety_flags, evidence)
        traces.append({"agent": consensus_agent.name, **c_out})
        return c_out["output"], traces

    output = merge_output(
        {
            "symptoms": symptoms,
            "risk_level": risk_level,
            "evidence": [],
            "response": f"Symptoms: {symptoms}. Risk: {risk_level}.",
            "uncertainty": "consensus skipped (ablation)",
            "safety_flags": safety_flags,
        },
        "",
    )
    traces.append({"agent": "consensus_skipped", "output": output, "raw_output": ""})
    return output, traces


def ablation_flags(ablation: str | None) -> dict[str, bool]:
    """Map ablation target to pipeline flags."""
    flags = {
        "enable_symptom": True,
        "enable_risk": True,
        "enable_safety": True,
        "enable_consensus": True,
    }
    if not ablation:
        return flags
    mapping = {
        "no_symptom_agent": "enable_symptom",
        "no_risk_agent": "enable_risk",
        "no_safety_agent": "enable_safety",
        "no_consensus_agent": "enable_consensus",
    }
    key = mapping.get(ablation)
    if key:
        flags[key] = False
    return flags
