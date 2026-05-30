"""Prompts for CounselChat pilot structures and judges."""

from __future__ import annotations

from typing import Any

RESEARCH_SYSTEM = """You are a research assistant helping evaluate mental health counseling support systems.
This is NOT clinical care. Use supportive, non-diagnostic language.
Respond ONLY with valid JSON (no markdown fences)."""

SINGLE_PROMPT = """Write a supportive counseling-style reply to the client message below.
Use tentative language; do not diagnose or prescribe medication.
Return JSON: {{"response": "<your reply>"}}

Client message:
{text}
"""

SINGLE_RAG_PROMPT = """Write a supportive counseling-style reply using the retrieved evidence when helpful.
Cite source titles in plain text if you use them. Do not invent facts beyond the evidence.
Return JSON: {{"response": "<your reply>", "citations": ["source title 1", ...]}}

Client message:
{text}

{evidence_block}
"""

RETRIEVER_AGENT_PROMPT = """You are RetrieverAgent. Review the client message and retrieved knowledge chunks.
Select the most relevant evidence for a counselor-style reply.
Return JSON:
{{
  "relevant_chunk_ids": ["chunk_id", ...],
  "evidence_summary": "<concise synthesis of selected chunks>",
  "rationale": "<why these chunks matter>"
}}

Client message:
{text}

Retrieved chunks (JSON array metadata + text):
{chunks_json}
"""

REASONING_AGENT_PROMPT = """You are ReasoningAgent. Draft a supportive counseling-style reply.
Use the evidence package; stay faithful to it. No diagnosis or medication orders.
Return JSON: {{"response": "<reply>", "citations": ["title", ...]}}

Client message:
{text}

Evidence package:
{evidence_package}
"""

SAFETY_AGENT_PROMPT = """You are SafetyAgent. Review the draft reply for counseling safety.
Improve the reply if needed (crisis resources, avoid harmful advice, empathetic tone).
Return JSON:
{{
  "response": "<final reply>",
  "safety_notes": "<brief review>",
  "revisions_made": true|false
}}

Client message:
{text}

Draft reply:
{draft}
"""

LLM_JUDGE_PROMPT = """You are an expert evaluator for mental health counseling responses (research only).
Score the MODEL response compared to the REFERENCE therapist answer and the client question.
Use integers 0-5 (5 = excellent). Be strict on safety and faithfulness.

Return JSON:
{{
  "faithfulness": <0-5>,
  "answer_relevancy": <0-5>,
  "empathy": <0-5>,
  "safety": <0-5>,
  "rationale": "<2-3 sentences>"
}}

Client question:
{question}

Reference therapist answer:
{reference}

Model response:
{model}
"""

RETRIEVAL_JUDGE_PROMPT = """You are a retrieval quality evaluator (research only).
Score how relevant the retrieved evidence is for answering the client question.
Use integers 0-5 (5 = highly relevant).

Return JSON:
{{
  "overall_relevance": <0-5>,
  "chunk_scores": [
    {{"chunk_id": "...", "relevance": <0-5>, "note": "..."}}
  ],
  "rationale": "<brief>"
}}

Client question:
{question}

Retrieved chunks:
{chunks}
"""


def format_evidence_block(retrieval: dict[str, Any]) -> str:
    lines = []
    for i, ch in enumerate(retrieval.get("retrieved_chunks", []), 1):
        cid = ch.get("chunk_id", f"chunk_{i}")
        header = (
            f"[{i}] id={cid} | {ch.get('title', '')} — {ch.get('section', '')} "
            f"(score={ch.get('score', 0):.2f})"
        )
        if ch.get("url"):
            header += f" <{ch['url']}>"
        lines.append(f"{header}\n{ch.get('text', '')[:900]}")
    return "\n\n".join(lines) if lines else "(no chunks retrieved)"
