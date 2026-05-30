"""Ollama HTTP client for local LLM inference."""

from __future__ import annotations

import logging
from typing import Any

import requests

from src.config import load_config
from src.utils import extract_json

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or load_config()
        ollama_cfg = cfg["ollama"]
        self.base_url = ollama_cfg["base_url"].rstrip("/")
        self.model = ollama_cfg["model"]
        self.timeout = ollama_cfg.get("timeout_seconds", 120)
        self.temperature = ollama_cfg.get("temperature", 0.1)

    def generate(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("response", "")
        except requests.RequestException as e:
            logger.error("Ollama request failed: %s", e)
            return {
                "raw_output": str(e),
                "parsed_json": None,
                "success": False,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        parsed = extract_json(raw)
        return {
            "raw_output": raw,
            "parsed_json": parsed,
            "success": True,
            "prompt_tokens": int(data.get("prompt_eval_count") or 0),
            "completion_tokens": int(data.get("eval_count") or 0),
        }

    def chat_json(self, user_prompt: str, system_prompt: str) -> dict[str, Any]:
        return self.generate(user_prompt, system=system_prompt)

    def with_model(self, model: str) -> OllamaClient:
        """Return a shallow copy bound to a different Ollama model tag."""
        import copy

        cloned = copy.copy(self)
        cloned.model = model
        return cloned
