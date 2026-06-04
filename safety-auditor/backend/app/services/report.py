from __future__ import annotations

import json
from typing import Any

from app.schemas import JudgeScores, ReportRequest, ResponsePayload


def _scores_table(plain: JudgeScores, agent: JudgeScores) -> str:
    rows = []
    for field in plain.model_fields:
        pv = getattr(plain, field)
        av = getattr(agent, field)
        rows.append(f"| {field} | {pv} | {av} | {round(av - pv, 1):+.1f} |")
    header = "| Metric | Single | Agent | Delta |\n| --- | ---: | ---: | ---: |"
    return header + "\n" + "\n".join(rows)


def generate_markdown(req: ReportRequest) -> str:
    ins = req.insights or "(insights not generated)"
    steps = "\n".join(
        f"- **{s.agent}** ({s.runtime_seconds}s): {s.output_summary}"
        for s in req.pipeline_steps
    )
    return f"""# Mental Health LLM Safety Auditor Report

## User Query
{req.query}

## Configuration
- Model: {req.model}
- Structure: {req.structure}

## Single LLM Response
{req.plain.text}

- Runtime: {req.plain.runtime_seconds}s
- Tokens: {req.plain.token_usage}

## Agent Structure Response
{req.agent.text}

- Runtime: {req.agent.runtime_seconds}s
- Tokens: {req.agent.token_usage}

## Pipeline
{steps}

## Judge Scores
{_scores_table(req.plain_scores, req.agent_scores)}

## Insights
{ins}
"""


def generate_json_report(req: ReportRequest) -> dict[str, Any]:
    return json.loads(req.model_dump_json())
