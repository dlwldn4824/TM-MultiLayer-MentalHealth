"""Route to Ollama or mock based on AUDITOR_LLM_MODE."""

from __future__ import annotations

import logging
from typing import List, Tuple

from app.config import LLM_MODE
from app.schemas import PipelineStep, ResponsePayload
from app.services import mock_llm, ollama_llm
from app.services.ollama_client import list_ollama_models

logger = logging.getLogger(__name__)


def use_ollama() -> bool:
    if LLM_MODE == "mock":
        return False
    if LLM_MODE == "ollama":
        return True
    return len(list_ollama_models()) > 0


async def generate_plain(query: str, model: str) -> ResponsePayload:
    if use_ollama():
        try:
            return await ollama_llm.generate_plain(query, model)
        except Exception as e:
            logger.warning("Ollama plain failed, fallback mock: %s", e)
            if LLM_MODE == "ollama":
                raise
    return await mock_llm.generate_plain(query, model)


async def generate_agent(
    query: str, model: str, structure: str
) -> Tuple[ResponsePayload, List[PipelineStep]]:
    if use_ollama():
        try:
            return await ollama_llm.generate_agent(query, model, structure)
        except Exception as e:
            logger.warning("Ollama agent failed, fallback mock: %s", e)
            if LLM_MODE == "ollama":
                raise
    return await mock_llm.generate_agent(query, model, structure)
