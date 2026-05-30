"""Sequential Experimental Design: Phase 0–6 with pruning between phases."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config import load_config
from src.data_loader import load_cases
from src.database import Database
from src.ensemble import combine_case_predictions
from src.evaluation import finalize_run, write_experiment_summary
from src.experiments import run_configured_experiment
from src.pruning import extract_model_winner, extract_structure_winner, pick_best, rank_runs
from src.run_config import RunConfig

logger = logging.getLogger(__name__)


def load_phase_state(path: str | None = None) -> dict[str, Any]:
    cfg = load_config()
    p = Path(path or cfg["output"]["phase_state_json"])
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_phase_state(state: dict[str, Any], path: str | None = None) -> None:
    cfg = load_config()
    p = Path(path or cfg["output"]["phase_state_json"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


class PhasePipeline:
    def __init__(self, config_path: str | None = None, only_phase: str | None = None):
        self.cfg = load_config(config_path)
        self.only_phase = only_phase
        self.phases_cfg = self.cfg.get("experiment_phases", {})
        self.pruning = self.cfg.get("pruning", {})
        self.state = load_phase_state()
        self.all_records: list[dict[str, Any]] = []
        self.db = Database(self.cfg["output"]["sqlite_path"])

    def _enabled(self, key: str) -> bool:
        if self.only_phase:
            return self.only_phase == key
        return bool(self.phases_cfg.get(key, False))

    def run(self) -> dict[str, Any]:
        if self._enabled("phase0_smoke_test"):
            self.run_phase0()
        if self._enabled("phase1_structure_comparison"):
            if self.state.get("phase0_passed", True):
                self.run_phase1()
        if self._enabled("phase2_model_comparison"):
            self.run_phase2()
        if self._enabled("phase3_ensemble_comparison"):
            self.run_phase3()
        if self._enabled("phase4_weight_search"):
            self.run_phase4()
        if self._enabled("phase5_ablation"):
            self.run_phase5()
        if self._enabled("phase6_data_scaling"):
            self.run_phase6()

        metrics_map = {r["experiment_name"]: r["metrics"] for r in self.all_records}
        save_path = self.cfg["output"]["metrics_json"]
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics_map, f, indent=2)

        write_experiment_summary(self.all_records, self.state)
        save_phase_state(self.state)
        return self.state

    def _execute_run(self, run_cfg: RunConfig, cases: list[dict]) -> dict[str, Any]:
        logger.info("Starting run: %s", run_cfg.experiment_name)
        results = run_configured_experiment(run_cfg, cases, self.cfg)
        record = finalize_run(run_cfg, results, self.all_records, self.db)
        return record

    def run_phase0(self) -> None:
        logger.info("=== Phase 0: Smoke Test ===")
        cases = load_cases(sample_size=10, force_csv=True)
        run_cfg = RunConfig(
            phase="phase0",
            structure="single_llm",
            model=self.cfg["ollama"]["model"],
            sample_size=10,
            force_csv=True,
            smoke=True,
        )
        record = self._execute_run(run_cfg, cases)
        m = record["metrics"]
        threshold = self.pruning.get("phase0_min_json_success", 0.5)
        passed = m.get("json_parse_success_rate", 0) >= threshold
        self.state["phase0_passed"] = passed
        self.state["phase0_metrics"] = m
        if not passed:
            logger.warning("Phase 0 failed JSON parse threshold (%.2f < %.2f)", m.get("json_parse_success_rate"), threshold)

    def run_phase1(self) -> None:
        logger.info("=== Phase 1: Structure Comparison ===")
        sample_size = self.cfg["data"].get("sample_size", 100)
        cases = load_cases(sample_size=sample_size, force_csv=False)
        model = self.cfg["ollama"]["model"]
        phase_records = []

        for structure in self.cfg.get("structures", self.cfg["experiments"]["names"]):
            run_cfg = RunConfig.from_structure(
                phase="phase1", structure=structure, model=model, sample_size=sample_size
            )
            phase_records.append(self._execute_run(run_cfg, cases))

        top_k = self.pruning.get("top_k_structures", 2)
        winners = rank_runs(phase_records, top_k=top_k)
        self.state["phase1_winners"] = [
            {
                "structure": extract_structure_winner(w),
                "experiment_name": w["experiment_name"],
                "metrics": w["metrics"],
            }
            for w in winners
        ]
        self.state["phase1_best_structure"] = extract_structure_winner(winners[0])

    def run_phase2(self) -> None:
        logger.info("=== Phase 2: Model Comparison ===")
        structure = self.state.get("phase1_best_structure", "multi_agent_rag")
        sample_size = self.cfg["data"].get("sample_size", 100)
        cases = load_cases(sample_size=sample_size, force_csv=False)
        phase_records = []

        for model in self.cfg.get("models", [self.cfg["ollama"]["model"]]):
            run_cfg = RunConfig.from_structure(
                phase="phase2", structure=structure, model=model, sample_size=sample_size
            )
            phase_records.append(self._execute_run(run_cfg, cases))

        top_k = self.pruning.get("top_k_models", 3)
        winners = rank_runs(phase_records, top_k=top_k)
        self.state["phase2_winners"] = [
            {"model": extract_model_winner(w), "experiment_name": w["experiment_name"], "metrics": w["metrics"]}
            for w in winners
        ]
        best = winners[0] if winners else pick_best(phase_records)
        if best:
            self.state["phase2_best_model"] = extract_model_winner(best)
            self.state["phase2_best_structure"] = structure

    def run_phase3(self) -> None:
        logger.info("=== Phase 3: Ensemble Comparison ===")
        structure = self.state.get("phase2_best_structure", self.state.get("phase1_best_structure", "multi_agent_rag"))
        sample_size = self.cfg["data"].get("sample_size", 100)
        cases = load_cases(sample_size=sample_size, force_csv=False)

        models = [w["model"] for w in self.state.get("phase2_winners", [])]
        if not models:
            models = self.cfg.get("ensemble_models_for_voting", self.cfg.get("models", []))[:3]

        model_metrics: dict[str, dict[str, float]] = {}
        per_case_by_model: dict[str, dict[str, dict[str, Any]]] = {m: {} for m in models}

        for model in models:
            run_cfg = RunConfig.from_structure(
                phase="phase3_base", structure=structure, model=model, sample_size=sample_size
            )
            results = run_configured_experiment(run_cfg, cases, self.cfg)
            model_metrics[model] = finalize_run(run_cfg, results, self.all_records, self.db)["metrics"]
            for r in results:
                per_case_by_model[model][r["case_id"]] = r["output"]

        phase_records = []
        for method in self.cfg.get("ensemble_methods", []):
            run_cfg = RunConfig(
                phase="phase3",
                structure=structure,
                model="ensemble",
                sample_size=sample_size,
                ensemble_method=method,
            )
            ens_results = self._build_ensemble_results(
                cases,
                per_case_by_model,
                models,
                method,
                model_metrics,
                experiment_name=run_cfg.experiment_name,
            )
            record = finalize_run(run_cfg, ens_results, self.all_records, self.db)
            phase_records.append(record)

        top_k = self.pruning.get("top_k_ensemble_configs", 2)
        winners = rank_runs(phase_records, top_k=top_k)
        self.state["phase3_winners"] = [
            {"ensemble_method": w["run_config"].ensemble_method, "metrics": w["metrics"]}
            for w in winners
            if isinstance(w.get("run_config"), RunConfig)
        ]
        if winners:
            rc = winners[0]["run_config"]
            self.state["phase3_best_ensemble"] = rc.ensemble_method if isinstance(rc, RunConfig) else None

    def run_phase4(self) -> None:
        logger.info("=== Phase 4: Weight Grid Search ===")
        structure = self.state.get("phase2_best_structure", "multi_agent_rag")
        sample_size = self.cfg["data"].get("sample_size", 100)
        cases = load_cases(sample_size=sample_size, force_csv=False)

        model_order = self.cfg.get("ensemble_models_for_voting", [])
        models = model_order[:3] if len(model_order) >= 3 else [
            w["model"] for w in self.state.get("phase2_winners", [])
        ][:3]
        if len(models) < 3:
            models = (models + self.cfg.get("models", []))[:3]

        per_case_by_model: dict[str, dict[str, dict[str, Any]]] = {m: {} for m in models}
        model_metrics: dict[str, dict[str, float]] = {}

        for model in models:
            run_cfg = RunConfig.from_structure(
                phase="phase4_base", structure=structure, model=model, sample_size=sample_size
            )
            results = run_configured_experiment(run_cfg, cases, self.cfg)
            model_metrics[model] = finalize_run(run_cfg, results, self.all_records, self.db)["metrics"]
            for r in results:
                per_case_by_model[model][r["case_id"]] = r["output"]

        phase_records = []
        for weights in self.cfg.get("weight_grid", []):
            run_cfg = RunConfig(
                phase="phase4",
                structure=structure,
                model="ensemble",
                sample_size=sample_size,
                ensemble_method="custom_weight",
                weights=weights,
            )
            ens_results = self._build_ensemble_results(
                cases,
                per_case_by_model,
                models,
                "custom_weight",
                model_metrics,
                experiment_name=run_cfg.experiment_name,
                custom_weights=weights,
            )
            phase_records.append(finalize_run(run_cfg, ens_results, self.all_records, self.db))

        best = pick_best(phase_records)
        if best:
            rc = best.get("run_config")
            self.state["phase4_best_weights"] = rc.weights if isinstance(rc, RunConfig) else None

    def run_phase5(self) -> None:
        logger.info("=== Phase 5: Agent / Layer Ablation ===")
        structure = self.state.get("phase2_best_structure", "multi_agent_rag")
        model = self.state.get("phase2_best_model", self.cfg["ollama"]["model"])
        sample_size = self.cfg["data"].get("sample_size", 100)
        cases = load_cases(sample_size=sample_size, force_csv=False)

        full_record = self._execute_run(
            RunConfig(
                phase="phase5",
                structure=structure,
                model=model,
                sample_size=sample_size,
                extra={"label": "full"},
            ),
            cases,
        )
        self.state["phase5_full"] = full_record["metrics"]

        for ablation in self.cfg.get("ablation_targets", []):
            if ablation == "no_retrieval":
                ab_structure = structure.replace("_rag", "") if "_rag" in structure else structure
                run_cfg = RunConfig(
                    phase="phase5",
                    structure=ab_structure,
                    model=model,
                    sample_size=sample_size,
                    ablation=ablation,
                )
            else:
                run_cfg = RunConfig(
                    phase="phase5",
                    structure=structure,
                    model=model,
                    sample_size=sample_size,
                    ablation=ablation,
                )
            self._execute_run(run_cfg, cases)

    def run_phase6(self) -> None:
        logger.info("=== Phase 6: Data Scaling ===")
        structure = self.state.get("phase2_best_structure", "multi_agent_rag")
        model = self.state.get("phase2_best_model", self.cfg["ollama"]["model"])
        scaling_results = []

        for sample_size in self.cfg.get("sample_sizes", [100, 300, 1000]):
            try:
                cases = load_cases(sample_size=sample_size, force_csv=False)
            except Exception as e:
                logger.warning("Could not load %d cases: %s", sample_size, e)
                continue
            run_cfg = RunConfig.from_structure(
                phase="phase6",
                structure=structure,
                model=model,
                sample_size=sample_size,
            )
            record = self._execute_run(run_cfg, cases)
            scaling_results.append(
                {"sample_size": sample_size, "metrics": record["metrics"], "experiment_name": record["experiment_name"]}
            )

        self.state["phase6_scaling"] = scaling_results

    def _build_ensemble_results(
        self,
        cases: list[dict],
        per_case_by_model: dict[str, dict[str, dict[str, Any]]],
        models: list[str],
        method: str,
        model_metrics: dict[str, dict[str, float]],
        experiment_name: str,
        custom_weights: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        results = []
        mname = "custom_weight" if custom_weights else method
        for case in cases:
            cid = str(case["case_id"])
            per_model = {m: per_case_by_model[m][cid] for m in models if cid in per_case_by_model.get(m, {})}
            if not per_model:
                continue
            output = combine_case_predictions(
                per_model,
                mname,
                model_metrics=model_metrics,
                custom_weights=custom_weights,
                model_order=models,
            )
            results.append(
                {
                    "case_id": cid,
                    "experiment_name": experiment_name,
                    "output": output,
                    "true_risk": case.get("risk_label"),
                    "true_symptoms": case.get("symptom_labels", []),
                    "hallucination_flag": False,
                    "unsafe_flag": False,
                    "json_parse_ok": True,
                    "runtime_sec": 0.0,
                }
            )
        return results
