"""Phase 2 analysis: tables, ranking, summary generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

MATRIX_COLUMNS = [
    "model",
    "structure",
    "bertscore_f1",
    "rouge_l",
    "judge_faithfulness",
    "judge_answer_relevancy",
    "judge_empathy",
    "judge_safety",
    "retrieval_relevance",
    "avg_runtime_sec",
    "token_throughput",
    "inference_stability",
    "json_success_rate",
    "n",
]

RANK_WEIGHTS = (
    ("judge_faithfulness", 5.0),
    ("judge_safety", 4.0),
    ("judge_empathy", 3.0),
    ("judge_answer_relevancy", 2.0),
    ("avg_runtime_sec", -0.05),  # lower is better
)


def experiment_key(model: str, structure: str) -> str:
    return f"phase2__{model.replace(':', '_').replace('.', '_')}__{structure}"


def flatten_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, m in metrics.items():
        if "|" in key:
            model, structure = key.split("|", 1)
        else:
            model, structure = m.get("model", ""), m.get("structure", key)
        rows.append(
            {
                "model": model,
                "structure": structure,
                "bertscore_f1": m.get("bertscore_f1", 0.0),
                "rouge_l": m.get("rouge_l", 0.0),
                "judge_faithfulness": m.get("judge_faithfulness", 0.0),
                "judge_answer_relevancy": m.get("judge_answer_relevancy", 0.0),
                "judge_empathy": m.get("judge_empathy", 0.0),
                "judge_safety": m.get("judge_safety", 0.0),
                "retrieval_relevance": m.get("retrieval_relevance", 0.0),
                "avg_runtime_sec": m.get("avg_runtime_sec", 0.0),
                "token_throughput": m.get("token_throughput", 0.0),
                "inference_stability": m.get("inference_stability", 0.0),
                "json_success_rate": m.get("json_success_rate", 0.0),
                "n": m.get("n", 0),
            }
        )
    return rows


def build_comparison_matrix(metrics: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(flatten_metrics(metrics))
    if df.empty:
        return pd.DataFrame(columns=MATRIX_COLUMNS)
    return df.sort_values(["model", "structure"]).reset_index(drop=True)


def rank_combinations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ranked = df.copy()
    ranked["rank_score"] = 0.0
    for col, weight in RANK_WEIGHTS:
        if col not in ranked.columns:
            continue
        vals = ranked[col].astype(float)
        if weight < 0:
            vals = vals.max() - vals
        ranked["rank_score"] += vals * abs(weight)
    ranked = ranked.sort_values("rank_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def _pair_label(model: str, structure: str) -> str:
    return f"{model} + {structure}"


def generate_analysis(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {f"q{i}": "Insufficient data." for i in range(1, 6)}

    rag = df[df["structure"] == "single_rag"]
    three = df[df["structure"] == "three_agent"]

    q1 = "Insufficient structure pairs."
    if not rag.empty and not three.empty:
        wins = 0
        metrics = ("judge_faithfulness", "judge_empathy", "judge_safety")
        for model in df["model"].unique():
            r = rag[rag["model"] == model]
            t = three[three["model"] == model]
            if r.empty or t.empty:
                continue
            r0, t0 = r.iloc[0], t.iloc[0]
            if all(t0.get(m, 0) >= r0.get(m, 0) for m in metrics):
                wins += 1
        total = len(set(rag["model"]) & set(three["model"]))
        q1 = (
            f"Three-Agent leads on faithfulness/empathy/safety vs single_rag in "
            f"{wins}/{total} models tested."
            if total
            else q1
        )

    q2 = "Need both qwen2.5:7b and qwen2.5:14b rows."
    q7 = df[df["model"] == "qwen2.5:7b"]
    q14 = df[df["model"] == "qwen2.5:14b"]
    if not q7.empty and not q14.empty:
        score7 = q7["judge_faithfulness"].mean() + q7["judge_safety"].mean()
        score14 = q14["judge_faithfulness"].mean() + q14["judge_safety"].mean()
        faster = q7["avg_runtime_sec"].mean() < q14["avg_runtime_sec"].mean()
        q2 = (
            f"14B aggregate faith+safety mean={score14:.2f} vs 7B={score7:.2f}. "
            f"{'14B wins on quality' if score14 > score7 else '7B matches or beats 14B on quality'}; "
            f"7B is {'faster' if faster else 'not faster'}."
        )

    small = df[df["model"].isin(["mistral:7b", "gemma2:9b"])]
    q3 = "No small-model rows yet."
    if not small.empty and not three.empty:
        gains = []
        for model in ["mistral:7b", "gemma2:9b"]:
            r = rag[rag["model"] == model]
            t = three[three["model"] == model]
            if r.empty or t.empty:
                continue
            delta = t.iloc[0]["judge_faithfulness"] - r.iloc[0]["judge_faithfulness"]
            gains.append(f"{model}: Δfaith={delta:+.2f}")
        q3 = (
            "Multi-Agent faithfulness delta vs single_rag: " + ", ".join(gains)
            if gains
            else q3
        )

    q4 = (
        f"Fastest avg runtime: {_pair_label(*df.loc[df['avg_runtime_sec'].idxmin(), ['model', 'structure']])} "
        f"({df['avg_runtime_sec'].min():.1f}s). "
        f"Highest faithfulness: {_pair_label(*df.loc[df['judge_faithfulness'].idxmax(), ['model', 'structure']])} "
        f"({df['judge_faithfulness'].max():.2f}/5)."
    )

    ranked = rank_combinations(df)
    best = ranked.iloc[0]
    q5 = f"Best overall (weighted rank): {_pair_label(best['model'], best['structure'])} (score={best['rank_score']:.2f})."

    return {
        "q1_three_agent_consistent": q1,
        "q2_qwen_14b_vs_7b": q2,
        "q3_small_models": q3,
        "q4_quality_latency": q4,
        "q5_best_pair": q5,
    }


def write_summary_md(
    path: str | Path,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    df: pd.DataFrame,
    ranked: pd.DataFrame,
) -> None:
    analysis = generate_analysis(df)
    lines = [
        "# Phase 2: Model Comparison Benchmark Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Setup",
        f"- Dataset: `{cfg['phase2']['dataset_id']}` (n={cfg['phase2'].get('sample_size', 30)})",
        f"- Structures: {', '.join(cfg['phase2']['structures'])}",
        f"- Models: {', '.join(cfg['phase2']['models'])}",
        "- Evaluation: BERTScore F1, ROUGE-L, LLM Judge (4 criteria), Retrieval Relevance",
        "- Runtime: avg seconds, token throughput, inference stability, JSON success rate",
        "",
        "## Table 1 — Structure × Model Quality Matrix",
        "",
        _matrix_markdown(df),
        "",
        "## Table 2 — Best Model Ranking",
        "",
        _ranking_markdown(ranked),
        "",
        "## Analysis",
        "",
        "### 1. Is Three-Agent consistently better across models?",
        analysis["q1_three_agent_consistent"],
        "",
        "### 2. Does qwen2.5:14b actually outperform qwen2.5:7b?",
        analysis["q2_qwen_14b_vs_7b"],
        "",
        "### 3. Do smaller models (mistral/gemma) preserve Multi-Agent gains?",
        analysis["q3_small_models"],
        "",
        "### 4. What is the quality–latency trade-off?",
        analysis["q4_quality_latency"],
        "",
        "### 5. Which model–structure pair is the best overall candidate?",
        analysis["q5_best_pair"],
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _matrix_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No results yet._"
    headers = [
        "Model",
        "Structure",
        "BERTScore",
        "ROUGE-L",
        "Faith",
        "Relev",
        "Empathy",
        "Safety",
        "Retrieval",
        "Runtime(s)",
        "JSON OK",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model']} | {r['structure']} | {r['bertscore_f1']:.3f} | {r['rouge_l']:.3f} | "
            f"{r['judge_faithfulness']:.2f} | {r['judge_answer_relevancy']:.2f} | "
            f"{r['judge_empathy']:.2f} | {r['judge_safety']:.2f} | "
            f"{r.get('retrieval_relevance', 0):.2f} | {r['avg_runtime_sec']:.1f} | "
            f"{r.get('json_success_rate', 0):.0%} |"
        )
    return "\n".join(lines)


def _ranking_markdown(ranked: pd.DataFrame) -> str:
    if ranked.empty:
        return "_No results yet._"
    lines = [
        "| Rank | Model | Structure | Score | Faith | Safety | Empathy | Relev | Runtime(s) |",
        "|------|-------|-----------|-------|-------|--------|---------|-------|------------|",
    ]
    for _, r in ranked.iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['model']} | {r['structure']} | {r['rank_score']:.2f} | "
            f"{r['judge_faithfulness']:.2f} | {r['judge_safety']:.2f} | "
            f"{r['judge_empathy']:.2f} | {r['judge_answer_relevancy']:.2f} | "
            f"{r['avg_runtime_sec']:.1f} |"
        )
    return "\n".join(lines)


def console_summary(df: pd.DataFrame, ranked: pd.DataFrame) -> str:
    if df.empty:
        return "Phase2 complete (no rows)."

    df2 = df.copy()
    df2["quality"] = (
        df2["judge_faithfulness"] + df2["judge_safety"] + df2["judge_empathy"]
    ) / 3.0
    best_quality = df2.loc[df2["quality"].idxmax()]
    best_speed = df.loc[df["avg_runtime_sec"].idxmin()]
    best_trade = ranked.iloc[0] if not ranked.empty else best_quality

    lines = [
        "Phase2 complete.",
        "",
        "Best quality:",
        f"  {best_quality['model']} + {best_quality['structure']}",
        "",
        "Best speed:",
        f"  {best_speed['model']} + {best_speed['structure']} ({best_speed['avg_runtime_sec']:.1f}s avg)",
        "",
        "Best tradeoff:",
        f"  {best_trade['model']} + {best_trade['structure']}",
    ]
    return "\n".join(lines)


def export_outputs(
    out: dict[str, str],
    metrics: dict[str, Any],
    df: pd.DataFrame,
    ranked: pd.DataFrame,
    judge_df: pd.DataFrame,
    retr_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
) -> None:
    Path(out["metrics_json"]).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    df.to_csv(out["metrics_csv"], index=False)
    df.to_csv(out["model_comparison_matrix_csv"], index=False)
    if not judge_df.empty:
        judge_df.to_csv(out["judge_scores_csv"], index=False)
    if not retr_df.empty:
        retr_df.to_csv(out["retrieval_scores_csv"], index=False)
    if not runtime_df.empty:
        runtime_df.to_csv(out["runtime_metrics_csv"], index=False)
