"""Reference metrics, LLM judge, and retrieval judge (no rule-based / pseudo-label eval)."""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from typing import Any

import pandas as pd
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer

from src.ollama_client import OllamaClient
from src.utils import extract_json

from src.pilot.io import parse_optional_json
from src.pilot.prompts import LLM_JUDGE_PROMPT, RESEARCH_SYSTEM, RETRIEVAL_JUDGE_PROMPT

logger = logging.getLogger(__name__)

JUDGE_CRITERIA = ("faithfulness", "answer_relevancy", "empathy", "safety")
RAG_STRUCTURES = frozenset({"single_rag", "two_agent", "three_agent", "conditional_bidirectional"})


def _clip_texts(texts: list[Any], max_chars: int = 1000) -> list[str]:
    """Truncate and sanitize for BERTScore (avoids tokenizer OverflowError on long texts)."""
    clipped: list[str] = []
    for t in texts:
        if t is None or (isinstance(t, float) and math.isnan(t)):
            s = ""
        else:
            s = str(t).strip()
        if not s:
            s = " "
        clipped.append(s[:max_chars])
    return clipped


def compute_reference_metrics(
    predictions: list[str],
    references: list[str],
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en",
    use_stemmer: bool = True,
) -> dict[str, float]:
    if not predictions:
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0, "rouge_l": 0.0}

    preds = _clip_texts(predictions)
    refs = _clip_texts(references)
    bert_f1 = 0.0
    bert_p = bert_r = 0.0
    try:
        logger.info(
            "BERTScore: encoding %d pairs with %s (may take 1–3 min, little console output)...",
            len(preds),
            bertscore_model,
        )
        P, R, F1 = bert_score_fn(
            preds,
            refs,
            model_type=bertscore_model,
            lang=lang,
            verbose=False,
            batch_size=8,
        )
        bert_p, bert_r, bert_f1 = float(P.mean()), float(R.mean()), float(F1.mean())
        logger.info(
            "BERTScore done: P=%.4f R=%.4f F1=%.4f",
            bert_p,
            bert_r,
            bert_f1,
        )
    except Exception as e:
        logger.warning("BERTScore failed (%s); ROUGE-L will still be computed.", e)

    predictions = preds
    references = refs
    logger.info("ROUGE-L: scoring %d pairs...", len(predictions))
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)
    rouge_vals = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge_vals.append(scores["rougeL"].fmeasure)
    rouge_mean = float(sum(rouge_vals) / len(rouge_vals))
    logger.info("ROUGE-L done: mean=%.4f", rouge_mean)

    return {
        "bertscore_precision": bert_p,
        "bertscore_recall": bert_r,
        "bertscore_f1": bert_f1,
        "rouge_l": rouge_mean,
    }


def _clamp_score(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(5.0, x))


def llm_judge_one(
    client: OllamaClient,
    question: str,
    reference: str,
    model_response: str,
    judge_temperature: float = 0.0,
) -> dict[str, Any]:
    prompt = LLM_JUDGE_PROMPT.format(
        question=question[:3000],
        reference=reference[:3000],
        model=model_response[:3000],
    )
    saved_temp = client.temperature
    client.temperature = judge_temperature
    out = client.chat_json(prompt, RESEARCH_SYSTEM)
    client.temperature = saved_temp
    parsed = out.get("parsed_json") or extract_json(out.get("raw_output", "")) or {}
    row = {k: _clamp_score(parsed.get(k)) for k in JUDGE_CRITERIA}
    row["rationale"] = str(parsed.get("rationale", ""))[:500]
    row["judge_raw"] = out.get("raw_output", "")[:300]
    return row


def retrieval_judge_one(
    client: OllamaClient,
    question: str,
    retrieval: dict[str, Any] | None,
    judge_temperature: float = 0.0,
) -> dict[str, Any]:
    chunks = (retrieval or {}).get("retrieved_chunks", [])
    if not chunks:
        return {
            "overall_relevance": 0.0,
            "chunk_scores_json": "[]",
            "rationale": "no_retrieval",
        }
    prompt = RETRIEVAL_JUDGE_PROMPT.format(
        question=question[:2500],
        chunks=json.dumps(chunks, ensure_ascii=False)[:10000],
    )
    saved_temp = client.temperature
    client.temperature = judge_temperature
    out = client.chat_json(prompt, RESEARCH_SYSTEM)
    client.temperature = saved_temp
    parsed = out.get("parsed_json") or extract_json(out.get("raw_output", "")) or {}
    return {
        "overall_relevance": _clamp_score(parsed.get("overall_relevance")),
        "chunk_scores_json": json.dumps(parsed.get("chunk_scores", []), ensure_ascii=False)[:4000],
        "rationale": str(parsed.get("rationale", ""))[:500],
    }


def evaluate_all(
    rows: list[dict[str, Any]],
    client: OllamaClient,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    ev = config.get("evaluation", {})
    ollama = config.get("ollama", {})
    judge_temp = float(ollama.get("judge_temperature", 0.0))

    judge_records: list[dict[str, Any]] = []
    retrieval_records: list[dict[str, Any]] = []
    metrics_by_structure: dict[str, Any] = {}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r["structure"]].append(r)

    run_llm = bool(ev.get("run_llm_judge", True))
    run_retr = bool(ev.get("run_retrieval_judge", True))
    logger.info(
        "Evaluation start: %d rows, structures=%s, llm_judge=%s, retrieval_judge=%s",
        len(rows),
        sorted(grouped.keys()),
        run_llm,
        run_retr,
    )

    for si, (structure, items) in enumerate(sorted(grouped.items()), start=1):
        logger.info(
            "[%d/%d] structure=%s (%d cases)",
            si,
            len(grouped),
            structure,
            len(items),
        )
        preds = [r["model_response"] for r in items]
        refs = [r["reference_answer"] for r in items]
        ref_m = compute_reference_metrics(
            preds,
            refs,
            bertscore_model=ev.get("bertscore_model", "microsoft/deberta-xlarge-mnli"),
            lang=ev.get("bertscore_lang", "en"),
            use_stemmer=bool(ev.get("rouge_use_stemmer", True)),
        )

        judge_rows = []
        if run_llm:
            n_judge = len(items)
            logger.info("LLM response judge: %s (%d cases, Ollama)...", structure, n_judge)
            for ji, r in enumerate(items, start=1):
                cid = r["case_id"]
                logger.info(
                    "  LLM judge [%s] case_id=%s (%d/%d)",
                    structure,
                    cid,
                    ji,
                    n_judge,
                )
                j = llm_judge_one(
                    client,
                    r["question_text"],
                    r["reference_answer"],
                    r["model_response"],
                    judge_temperature=judge_temp,
                )
                rec = {
                    "case_id": cid,
                    "structure": structure,
                    **{k: j[k] for k in JUDGE_CRITERIA},
                    "judge_rationale": j.get("rationale", ""),
                }
                judge_records.append(rec)
                judge_rows.append(j)
                logger.info(
                    "  LLM judge done case_id=%s faith=%.1f rel=%.1f emp=%.1f safe=%.1f",
                    cid,
                    j["faithfulness"],
                    j["answer_relevancy"],
                    j["empathy"],
                    j["safety"],
                )

        retr_rows = []
        if run_retr and structure in RAG_STRUCTURES:
            n_retr = len(items)
            logger.info("Retrieval judge: %s (%d cases)...", structure, n_retr)
            for ri, r in enumerate(items, start=1):
                cid = r["case_id"]
                logger.info(
                    "  Retrieval judge [%s] case_id=%s (%d/%d)",
                    structure,
                    cid,
                    ri,
                    n_retr,
                )
                retrieval = r.get("retrieval")
                if not isinstance(retrieval, dict):
                    retrieval = parse_optional_json(
                        retrieval or r.get("retrieval_json"),
                        ctx=f"eval {r.get('case_id')}",
                    )
                tr = retrieval_judge_one(
                    client, r["question_text"], retrieval, judge_temperature=judge_temp
                )
                rec = {
                    "case_id": cid,
                    "structure": structure,
                    "overall_relevance": tr["overall_relevance"],
                    "chunk_scores_json": tr["chunk_scores_json"],
                    "retrieval_rationale": tr.get("rationale", ""),
                }
                retrieval_records.append(rec)
                retr_rows.append(tr)
                logger.info(
                    "  Retrieval judge done case_id=%s relevance=%.1f",
                    cid,
                    tr["overall_relevance"],
                )

        agg = dict(ref_m)
        agg["n"] = len(items)
        agg["avg_runtime_sec"] = sum(r.get("runtime_sec", 0) for r in items) / len(items)
        if judge_rows:
            for k in JUDGE_CRITERIA:
                agg[f"judge_{k}"] = sum(j[k] for j in judge_rows) / len(judge_rows)
        if retr_rows:
            agg["retrieval_relevance"] = sum(t["overall_relevance"] for t in retr_rows) / len(retr_rows)

        metrics_by_structure[structure] = agg
        logger.info(
            "Structure %s summary: bertscore_f1=%.4f rouge_l=%.4f n=%d",
            structure,
            agg.get("bertscore_f1", 0.0),
            agg.get("rouge_l", 0.0),
            agg.get("n", 0),
        )

    logger.info(
        "Evaluation complete: %d structures, %d judge rows, %d retrieval rows",
        len(metrics_by_structure),
        len(judge_records),
        len(retrieval_records),
    )
    return metrics_by_structure, pd.DataFrame(judge_records), pd.DataFrame(retrieval_records)
