"""Experiment run configuration (structure, model, phase metadata)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


STRUCTURES = ("single_llm", "single_llm_rag", "multi_agent", "multi_agent_rag")


@dataclass
class RunConfig:
    phase: str
    structure: str
    model: str
    sample_size: int
    use_rag: bool = False
    ablation: str | None = None
    ensemble_method: str | None = None
    weights: list[float] | None = None
    dataset_name: str = "ShenLab/MentalChat16K"
    force_csv: bool = False
    smoke: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.use_rag = self.structure in ("single_llm_rag", "multi_agent_rag")
        if self.ablation == "no_retrieval":
            self.use_rag = False

    @property
    def experiment_name(self) -> str:
        parts = [self.phase, self.structure, self._model_slug(), f"n{self.sample_size}"]
        if self.ablation:
            parts.append(self.ablation)
        if self.ensemble_method:
            parts.append(self.ensemble_method)
        if self.weights:
            w = "_".join(f"{x:.1f}" for x in self.weights)
            parts.append(f"w{w}")
        if self.smoke:
            parts.append("smoke")
        return "__".join(parts)

    def _model_slug(self) -> str:
        return self.model.replace(":", "_").replace(".", "_")

    @property
    def knowledge_base_mode(self) -> str:
        from src.config import load_config

        return load_config().get("knowledge_base", {}).get("mode", "official")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["experiment_name"] = self.experiment_name
        d["use_rag"] = self.use_rag
        d["knowledge_base_mode"] = self.knowledge_base_mode
        return d

    @classmethod
    def from_structure(
        cls,
        phase: str,
        structure: str,
        model: str,
        sample_size: int,
        **kwargs: Any,
    ) -> RunConfig:
        return cls(phase=phase, structure=structure, model=model, sample_size=sample_size, **kwargs)
