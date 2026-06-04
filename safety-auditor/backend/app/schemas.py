from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JudgeScores(BaseModel):
    safety: float = Field(ge=1, le=5)
    empathy: float = Field(ge=1, le=5)
    trust: float = Field(ge=1, le=5)
    helpfulness: float = Field(ge=1, le=5)
    faithfulness: float = Field(ge=1, le=5)
    coherence: float = Field(ge=1, le=5)


class PipelineStep(BaseModel):
    agent: str
    runtime_seconds: float
    prompt_summary: str
    output_summary: str


class CompareRequest(BaseModel):
    query: str
    model: str = "qwen2.5:7b"
    structure: str = "emotion_strategy_response_safety"
    judge_provider: str = "llama3.1:8b"


class ResponsePayload(BaseModel):
    text: str
    runtime_seconds: float
    token_usage: Dict[str, int]


class CompareResponse(BaseModel):
    query: str
    model: str
    structure: str
    plain: ResponsePayload
    agent: ResponsePayload
    pipeline_steps: List[PipelineStep]
    plain_scores: JudgeScores
    agent_scores: JudgeScores
    evaluation_id: Optional[int] = None


class InsightRequest(BaseModel):
    query: str
    model: str
    structure: str
    plain_scores: JudgeScores
    agent_scores: JudgeScores
    runtime_plain: float
    runtime_agent: float


class InsightResponse(BaseModel):
    summary: str
    bullets: List[str]


class ReportRequest(BaseModel):
    query: str
    model: str
    structure: str
    plain: ResponsePayload
    agent: ResponsePayload
    pipeline_steps: List[PipelineStep]
    plain_scores: JudgeScores
    agent_scores: JudgeScores
    insights: Optional[str] = None
    format: str = "markdown"  # markdown | json


class CsvUploadResponse(BaseModel):
    upload_id: int
    label: str
    row_count: int
    summary: dict[str, Any]
    chart_data: list[dict[str, Any]]
