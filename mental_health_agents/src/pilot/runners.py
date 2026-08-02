"""Run pilot inference structures (Single / +RAG / 2-Agent / 3-Agent)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.ollama_client import OllamaClient
from src.utils import extract_json

from src.pilot.kb import PilotRAG
from src.pilot.prompts import (
    REASONING_AGENT_PROMPT,
    RESEARCH_SYSTEM,
    RETRIEVER_AGENT_PROMPT,
    REVISION_AGENT_PROMPT,
    SAFETY_AGENT_PROMPT,
    SAFETY_GATEKEEPER_PROMPT,
    SAFETY_RECHECK_PROMPT,
    SINGLE_PROMPT,
    SINGLE_RAG_PROMPT,
    format_evidence_block,
)

logger = logging.getLogger(__name__)

STRUCTURE_SINGLE = "single"
STRUCTURE_SINGLE_RAG = "single_rag"
STRUCTURE_TWO_AGENT = "two_agent"
STRUCTURE_THREE_AGENT = "three_agent"
STRUCTURE_CONDITIONAL_BIDIRECTIONAL = "conditional_bidirectional"

ALL_STRUCTURES = (
    STRUCTURE_SINGLE,
    STRUCTURE_SINGLE_RAG,
    STRUCTURE_TWO_AGENT,
    STRUCTURE_THREE_AGENT,
    STRUCTURE_CONDITIONAL_BIDIRECTIONAL,
)


def _llm_json(client: OllamaClient, prompt: str) -> tuple[dict[str, Any], str]:
    out = client.chat_json(prompt, RESEARCH_SYSTEM)
    parsed = out.get("parsed_json") or {}
    if not isinstance(parsed, dict):
        parsed = {}
    return parsed, out.get("raw_output", "")


def _response_text(parsed: dict[str, Any], raw: str) -> str:
    if parsed.get("response"):
        return str(parsed["response"]).strip()
    return raw.strip()[:2000]


def run_single(client: OllamaClient, question: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    parsed, raw = _llm_json(client, SINGLE_PROMPT.format(text=question))
    return {
        "structure": STRUCTURE_SINGLE,
        "response": _response_text(parsed, raw),
        "agent_trace": [],
        "retrieval": None,
        "runtime_sec": time.perf_counter() - t0,
    }


def run_single_rag(client: OllamaClient, rag: PilotRAG, question: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    retrieval = rag.retrieve_structured(question)
    prompt = SINGLE_RAG_PROMPT.format(
        text=question,
        evidence_block=format_evidence_block(retrieval),
    )
    parsed, raw = _llm_json(client, prompt)
    return {
        "structure": STRUCTURE_SINGLE_RAG,
        "response": _response_text(parsed, raw),
        "agent_trace": [],
        "retrieval": retrieval,
        "runtime_sec": time.perf_counter() - t0,
    }


def run_two_agent(client: OllamaClient, rag: PilotRAG, question: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    retrieval = rag.retrieve_structured(question)
    chunks_json = json.dumps(retrieval.get("retrieved_chunks", []), ensure_ascii=False)[:12000]

    ret_parsed, ret_raw = _llm_json(
        client,
        RETRIEVER_AGENT_PROMPT.format(text=question, chunks_json=chunks_json),
    )
    evidence_package = {
        "relevant_chunk_ids": ret_parsed.get("relevant_chunk_ids", []),
        "evidence_summary": ret_parsed.get("evidence_summary", ""),
        "rationale": ret_parsed.get("rationale", ""),
        "retrieval": retrieval,
    }
    if not evidence_package["evidence_summary"]:
        evidence_package["evidence_summary"] = format_evidence_block(retrieval)

    reason_parsed, reason_raw = _llm_json(
        client,
        REASONING_AGENT_PROMPT.format(
            text=question,
            evidence_package=json.dumps(evidence_package, ensure_ascii=False)[:8000],
        ),
    )
    trace = [
        {"agent": "RetrieverAgent", "output": ret_parsed, "raw": ret_raw[:500]},
        {"agent": "ReasoningAgent", "output": reason_parsed, "raw": reason_raw[:500]},
    ]
    return {
        "structure": STRUCTURE_TWO_AGENT,
        "response": _response_text(reason_parsed, reason_raw),
        "agent_trace": trace,
        "retrieval": retrieval,
        "runtime_sec": time.perf_counter() - t0,
    }


def run_conditional_bidirectional(
    client: OllamaClient, rag: PilotRAG, question: str
) -> dict[str, Any]:
    """Question → Retrieval → Response → Safety Gatekeeper → [Revision → Recheck] → Final."""
    t0 = time.perf_counter()
    retrieval = rag.retrieve_structured(question)

    resp_parsed, resp_raw = _llm_json(
        client,
        SINGLE_RAG_PROMPT.format(
            text=question,
            evidence_block=format_evidence_block(retrieval),
        ),
    )
    draft = _response_text(resp_parsed, resp_raw)

    gate_parsed, gate_raw = _llm_json(
        client,
        SAFETY_GATEKEEPER_PROMPT.format(text=question, draft=draft),
    )
    revision_needed = bool(gate_parsed.get("revision_needed"))
    safety_pass = gate_parsed.get("safety_pass")
    if safety_pass is True or safety_pass == "true":
        revision_needed = False
    elif safety_pass is False or safety_pass == "false":
        revision_needed = True

    trace: list[dict[str, Any]] = [
        {"agent": "Retrieval", "output": {"chunk_count": len(retrieval.get("retrieved_chunks", []))}},
        {"agent": "ResponseAgent", "output": resp_parsed, "raw": resp_raw[:500]},
        {"agent": "SafetyGatekeeper", "output": gate_parsed, "raw": gate_raw[:500]},
    ]

    llm_calls = 3
    revision_triggered = False
    final = draft

    if revision_needed:
        revision_triggered = True
        issues = gate_parsed.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        rev_parsed, rev_raw = _llm_json(
            client,
            REVISION_AGENT_PROMPT.format(
                text=question,
                draft=draft,
                issues=json.dumps(issues, ensure_ascii=False),
                rationale=str(gate_parsed.get("rationale", ""))[:1500],
            ),
        )
        llm_calls += 1
        revised = _response_text(rev_parsed, rev_raw) or draft
        trace.append({"agent": "RevisionAgent", "output": rev_parsed, "raw": rev_raw[:500]})

        recheck_parsed, recheck_raw = _llm_json(
            client,
            SAFETY_RECHECK_PROMPT.format(text=question, revised=revised),
        )
        llm_calls += 1
        final = _response_text(recheck_parsed, recheck_raw) or revised
        trace.append({"agent": "SafetyRecheck", "output": recheck_parsed, "raw": recheck_raw[:500]})
    else:
        trace.append({"agent": "RevisionAgent", "output": {"skipped": True}})
        trace.append({"agent": "SafetyRecheck", "output": {"skipped": True}})

    return {
        "structure": STRUCTURE_CONDITIONAL_BIDIRECTIONAL,
        "response": final,
        "agent_trace": trace,
        "retrieval": retrieval,
        "runtime_sec": time.perf_counter() - t0,
        "llm_calls": llm_calls,
        "revision_triggered": revision_triggered,
        "gatekeeper_pass": not revision_needed,
    }


def run_three_agent(client: OllamaClient, rag: PilotRAG, question: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    retrieval = rag.retrieve_structured(question)
    chunks_json = json.dumps(retrieval.get("retrieved_chunks", []), ensure_ascii=False)[:12000]

    ret_parsed, ret_raw = _llm_json(
        client,
        RETRIEVER_AGENT_PROMPT.format(text=question, chunks_json=chunks_json),
    )
    evidence_package = {
        "relevant_chunk_ids": ret_parsed.get("relevant_chunk_ids", []),
        "evidence_summary": ret_parsed.get("evidence_summary", "") or format_evidence_block(retrieval),
        "rationale": ret_parsed.get("rationale", ""),
    }

    reason_parsed, reason_raw = _llm_json(
        client,
        REASONING_AGENT_PROMPT.format(
            text=question,
            evidence_package=json.dumps(evidence_package, ensure_ascii=False)[:8000],
        ),
    )
    draft = _response_text(reason_parsed, reason_raw)

    safety_parsed, safety_raw = _llm_json(
        client,
        SAFETY_AGENT_PROMPT.format(text=question, draft=draft),
    )
    final = _response_text(safety_parsed, safety_raw) or draft

    trace = [
        {"agent": "RetrieverAgent", "output": ret_parsed},
        {"agent": "ReasoningAgent", "output": reason_parsed},
        {"agent": "SafetyAgent", "output": safety_parsed},
    ]
    return {
        "structure": STRUCTURE_THREE_AGENT,
        "response": final,
        "agent_trace": trace,
        "retrieval": retrieval,
        "runtime_sec": time.perf_counter() - t0,
    }


def run_structure(
    structure: str,
    client: OllamaClient,
    rag: PilotRAG | None,
    case: dict[str, Any],
) -> dict[str, Any]:
    question = case["question_text"]
    if structure == STRUCTURE_SINGLE:
        result = run_single(client, question)
    elif structure == STRUCTURE_SINGLE_RAG:
        if rag is None:
            raise ValueError("RAG required for single_rag")
        result = run_single_rag(client, rag, question)
    elif structure == STRUCTURE_TWO_AGENT:
        if rag is None:
            raise ValueError("RAG required for two_agent")
        result = run_two_agent(client, rag, question)
    elif structure == STRUCTURE_THREE_AGENT:
        if rag is None:
            raise ValueError("RAG required for three_agent")
        result = run_three_agent(client, rag, question)
    elif structure == STRUCTURE_CONDITIONAL_BIDIRECTIONAL:
        if rag is None:
            raise ValueError("RAG required for conditional_bidirectional")
        result = run_conditional_bidirectional(client, rag, question)
    else:
        raise ValueError(f"Unknown structure: {structure}")

    result["case_id"] = case["case_id"]
    result["question_text"] = question
    result["reference_answer"] = case["reference_answer"]
    result["topic"] = case.get("topic", "")
    return result
