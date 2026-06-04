"""Local Ollama client (no OpenAI)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

import requests

from app.config import OLLAMA_BASE_URL, OLLAMA_NUM_CTX, OLLAMA_TEMPERATURE, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        try:
            obj = json.loads(brace.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


class OllamaClient:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = OLLAMA_TIMEOUT,
        temperature: float = OLLAMA_TEMPERATURE,
        num_ctx: int = OLLAMA_NUM_CTX,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx

    def generate(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if system:
            payload["system"] = system
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.exception("Ollama generate failed model=%s", self.model)
            return {"success": False, "raw_output": "", "parsed_json": None, "error": str(e)}
        raw = str(data.get("response") or "").strip()
        return {
            "success": True,
            "raw_output": raw,
            "parsed_json": extract_json(raw),
            "prompt_tokens": int(data.get("prompt_eval_count") or 0),
            "completion_tokens": int(data.get("eval_count") or 0),
        }

    def chat_json(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        return self.generate(prompt, system=system)


def list_ollama_models() -> list[str]:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return [m for m in models if m]
    except Exception as e:
        logger.warning("Could not list Ollama models: %s", e)
        return []
