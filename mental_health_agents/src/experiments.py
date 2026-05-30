"""Run experiment conditions with RunConfig metadata and timing."""

from __future__ import annotations

import logging
import time
from typing import Any

from tqdm import tqdm

from src.agents import ablation_flags, run_multi_agent, run_single_llm
from src.config import load_config
from src.database import Database
from src.ollama_client import OllamaClient
from src.rag import KnowledgeRAG, evidence_texts_from_retrieval
from src.run_config import RunConfig

logger = logging.getLogger(__name__)

EXPERIMENT_SINGLE = "single_llm"
EXPERIMENT_SINGLE_RAG = "single_llm_rag"
EXPERIMENT_MULTI = "multi_agent"
EXPERIMENT_MULTI_RAG = "multi_agent_rag"


def _any_safety_flag(output: dict[str, Any]) -> bool:
    flags = output.get("safety_flags", {})
    return any(bool(v) for v in flags.values()) if isinstance(flags, dict) else False


def _hallucination_flag(output: dict[str, Any]) -> bool:
    flags = output.get("safety_flags", {})
    if not isinstance(flags, dict):
        return False
    return bool(flags.get("fabricated_symptom") or flags.get("unsupported_diagnosis"))


def _structure_from_name(name: str) -> str:
    return name


def run_case_with_config(
    run_cfg: RunConfig,
    case: dict[str, Any],
    client: OllamaClient,
    rag: KnowledgeRAG | None,
    db: Database,
) -> dict[str, Any]:
    experiment_name = run_cfg.experiment_name
    case_id = str(case["case_id"])
    text = case["text"]
    structure = run_cfg.structure

    retrieval_payload: dict[str, Any] | None = None
    evidence: list[str] = []
    use_rag = run_cfg.use_rag and run_cfg.ablation != "no_retrieval"
    if use_rag and rag:
        retrieval_payload = rag.retrieve_structured(text)
        evidence = evidence_texts_from_retrieval(retrieval_payload)

    t0 = time.perf_counter()
    json_ok = False

    if structure in (EXPERIMENT_SINGLE, EXPERIMENT_SINGLE_RAG):
        result = run_single_llm(client, text, evidence if use_rag else None)
        output = result["output"]
        json_ok = result.get("parsed_json") is not None
        db.save_agent_output(
            case_id,
            experiment_name,
            "single_llm",
            output,
            result["raw_output"],
            evidence,
            retrieval=retrieval_payload,
        )
    else:
        flags = ablation_flags(run_cfg.ablation)
        output, traces = run_multi_agent(
            client, text, evidence if use_rag else None, **flags
        )
        json_ok = any(t.get("parsed") for t in traces)
        for trace in traces:
            agent = trace.get("agent", "unknown")
            parsed = trace.get("parsed") or trace.get("output")
            raw = trace.get("raw_output", "")
            db.save_agent_output(
                case_id, experiment_name, agent, parsed, raw, evidence, retrieval=retrieval_payload
            )

    elapsed = time.perf_counter() - t0
    unsafe = _any_safety_flag(output)
    halluc = _hallucination_flag(output)

    db.save_prediction(
        case_id=case_id,
        experiment_name=experiment_name,
        predicted_risk=output.get("risk_level", "low"),
        predicted_symptoms=output.get("symptoms", []),
        final_response=output.get("response", ""),
        hallucination_flag=halluc,
        unsafe_flag=unsafe,
        run_metadata=run_cfg.to_dict(),
        retrieval=retrieval_payload,
    )

    return {
        "case_id": case_id,
        "experiment_name": experiment_name,
        "run_config": run_cfg,
        "output": output,
        "true_risk": case.get("risk_label"),
        "true_symptoms": case.get("symptom_labels", []),
        "hallucination_flag": halluc,
        "unsafe_flag": unsafe,
        "json_parse_ok": json_ok,
        "runtime_sec": elapsed,
        "retrieval": retrieval_payload,
        "use_rag": use_rag,
    }


def run_configured_experiment(
    run_cfg: RunConfig,
    cases: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = config or load_config()
    cfg = dict(cfg)
    cfg["ollama"] = {**cfg["ollama"], "model": run_cfg.model}

    client = OllamaClient(cfg)
    db = Database(cfg["output"]["sqlite_path"])
    rag = KnowledgeRAG(cfg) if run_cfg.use_rag else None

    results: list[dict[str, Any]] = []
    for case in cases:
        db.upsert_case(case)

    desc = run_cfg.experiment_name
    for case in tqdm(cases, desc=desc):
        results.append(run_case_with_config(run_cfg, case, client, rag, db))

    return results


# Backward-compatible API
def run_experiment_on_case(
    experiment_name: str,
    case: dict[str, Any],
    client: OllamaClient,
    rag: KnowledgeRAG | None,
    db: Database,
) -> dict[str, Any]:
    run_cfg = RunConfig(
        phase="legacy",
        structure=_structure_from_name(experiment_name),
        model=client.model,
        sample_size=1,
    )
    return run_case_with_config(run_cfg, case, client, rag, db)


def run_all_experiments(
    cases: list[dict[str, Any]],
    experiment_names: list[str] | None = None,
    config_path: str | None = None,
) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    names = experiment_names or cfg["experiments"]["names"]
    all_results: list[dict[str, Any]] = []
    for name in names:
        run_cfg = RunConfig.from_structure(
            phase="legacy",
            structure=name,
            model=cfg["ollama"]["model"],
            sample_size=len(cases),
        )
        all_results.extend(run_configured_experiment(run_cfg, cases, cfg))
    return all_results
