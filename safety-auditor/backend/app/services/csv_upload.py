"""Parse experiment CSV uploads and aggregate metrics."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

METRIC_COLS = [
    "safety",
    "empathy",
    "trust",
    "helpfulness",
    "faithfulness",
    "coherence",
    "relevance",
    "fluency",
    "runtime_seconds",
    "final_score",
]


def parse_csv_content(content: bytes, label: str) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    df = pd.read_csv(io.BytesIO(content))
    row_count = len(df)
    summary: dict[str, Any] = {"label": label, "n": row_count}
    for col in METRIC_COLS:
        if col in df.columns:
            summary[f"mean_{col}"] = float(pd.to_numeric(df[col], errors="coerce").mean())

    chart_data = []
    for col in ("safety", "empathy", "trust", "helpfulness", "faithfulness"):
        if col in df.columns:
            chart_data.append(
                {
                    "metric": col,
                    "value": float(pd.to_numeric(df[col], errors="coerce").mean()),
                    "label": label,
                }
            )
    return row_count, summary, chart_data


def merge_chart_data(uploads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by metric for multi-series charts."""
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for u in uploads:
        label = u.get("label", "unknown")
        for pt in u.get("chart_data", []):
            m = pt["metric"]
            by_metric.setdefault(m, []).append(
                {"label": label, "value": pt["value"], "metric": m}
            )
    out = []
    for metric, series in by_metric.items():
        out.append({"metric": metric, "series": series})
    return out
