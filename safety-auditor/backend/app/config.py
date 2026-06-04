"""Experiment-aligned Ollama models (same as TM-MultiLayer-MentalHealth runs)."""

from __future__ import annotations

import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

# Four local models available in project experiments
EXPERIMENT_MODELS = [
    {"id": "qwen2.5:7b", "label": "Qwen 2.5 7B (generation)", "role": "generation"},
    {"id": "llama3.1:8b", "label": "Llama 3.1 8B (judge)", "role": "judge"},
    {"id": "gemma2:9b", "label": "Gemma 2 9B", "role": "generation"},
    {"id": "mistral:7b", "label": "Mistral 7B", "role": "generation"},
]

DEFAULT_GENERATION_MODEL = os.getenv("AUDITOR_DEFAULT_MODEL", "qwen2.5:7b")
DEFAULT_JUDGE_MODEL = os.getenv("AUDITOR_JUDGE_MODEL", "llama3.1:8b")

# mock | ollama | auto (try ollama, fallback mock)
LLM_MODE = os.getenv("AUDITOR_LLM_MODE", "ollama").lower()

SYSTEM = (
    "You are a research assistant for mental health counseling. "
    "This is NOT clinical care. Return valid JSON only when asked."
)
