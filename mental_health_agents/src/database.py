"""SQLite persistence for experiments."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from src.config import load_config
from src.utils import ensure_dir


class Database:
    def __init__(self, db_path: str | None = None):
        cfg = load_config()
        self.db_path = db_path or cfg["output"]["sqlite_path"]
        ensure_dir(self.db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT,
                    risk_label TEXT,
                    symptom_labels TEXT
                );

                CREATE TABLE IF NOT EXISTS experiment_runs (
                    experiment_name TEXT PRIMARY KEY,
                    phase TEXT,
                    model TEXT,
                    sample_size INTEGER,
                    structure TEXT,
                    use_rag INTEGER,
                    knowledge_base_mode TEXT,
                    ablation TEXT,
                    ensemble_method TEXT,
                    weights TEXT,
                    dataset_name TEXT,
                    metrics_json TEXT,
                    json_parse_success_rate REAL,
                    avg_runtime_sec REAL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    experiment_name TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    output_json TEXT,
                    raw_output TEXT,
                    evidence TEXT,
                    retrieval_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    experiment_name TEXT NOT NULL,
                    predicted_risk TEXT,
                    predicted_symptoms TEXT,
                    final_response TEXT,
                    hallucination_flag INTEGER DEFAULT 0,
                    unsafe_flag INTEGER DEFAULT 0,
                    model TEXT,
                    sample_size INTEGER,
                    use_rag INTEGER,
                    structure TEXT,
                    ablation TEXT,
                    ensemble_method TEXT,
                    weights TEXT,
                    phase TEXT,
                    UNIQUE(case_id, experiment_name)
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    UNIQUE(experiment_name, metric_name)
                );
                """
            )
            self._migrate_predictions_columns(conn)

    def _migrate_predictions_columns(self, conn: sqlite3.Connection) -> None:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
        for col, typ in [
            ("model", "TEXT"),
            ("sample_size", "INTEGER"),
            ("use_rag", "INTEGER"),
            ("structure", "TEXT"),
            ("ablation", "TEXT"),
            ("ensemble_method", "TEXT"),
            ("weights", "TEXT"),
            ("phase", "TEXT"),
            ("knowledge_base_mode", "TEXT"),
            ("retrieval_json", "TEXT"),
        ]:
            if col not in cols:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")

        ao_cols = {row[1] for row in conn.execute("PRAGMA table_info(agent_outputs)").fetchall()}
        if "retrieval_json" not in ao_cols:
            conn.execute("ALTER TABLE agent_outputs ADD COLUMN retrieval_json TEXT")

        er_cols = {row[1] for row in conn.execute("PRAGMA table_info(experiment_runs)").fetchall()}
        if "knowledge_base_mode" not in er_cols:
            conn.execute("ALTER TABLE experiment_runs ADD COLUMN knowledge_base_mode TEXT")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS phase2_counsel_benchmark (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phase TEXT NOT NULL DEFAULT 'phase2',
                model TEXT NOT NULL,
                structure TEXT NOT NULL,
                case_id TEXT,
                runtime_sec REAL,
                bertscore_f1 REAL,
                rouge_l REAL,
                judge_faithfulness REAL,
                judge_answer_relevancy REAL,
                judge_empathy REAL,
                judge_safety REAL,
                retrieval_relevance REAL,
                json_success_rate REAL,
                token_throughput REAL,
                inference_success INTEGER DEFAULT 1,
                experiment_key TEXT NOT NULL,
                metrics_json TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(experiment_key, case_id)
            )
            """
        )

    def upsert_case(self, case: dict[str, Any]) -> None:
        symptoms = json.dumps(case.get("symptom_labels", []))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cases (case_id, text, source, risk_label, symptom_labels)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    text=excluded.text,
                    source=excluded.source,
                    risk_label=excluded.risk_label,
                    symptom_labels=excluded.symptom_labels
                """,
                (
                    str(case["case_id"]),
                    case["text"],
                    case.get("source", ""),
                    case.get("risk_label", ""),
                    symptoms,
                ),
            )

    def save_experiment_run(
        self,
        experiment_name: str,
        metadata: dict[str, Any],
        metrics: dict[str, float],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiment_runs (
                    experiment_name, phase, model, sample_size, structure, use_rag,
                    knowledge_base_mode, ablation, ensemble_method, weights, dataset_name,
                    metrics_json, json_parse_success_rate, avg_runtime_sec, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_name) DO UPDATE SET
                    phase=excluded.phase,
                    model=excluded.model,
                    sample_size=excluded.sample_size,
                    structure=excluded.structure,
                    use_rag=excluded.use_rag,
                    knowledge_base_mode=excluded.knowledge_base_mode,
                    ablation=excluded.ablation,
                    ensemble_method=excluded.ensemble_method,
                    weights=excluded.weights,
                    dataset_name=excluded.dataset_name,
                    metrics_json=excluded.metrics_json,
                    json_parse_success_rate=excluded.json_parse_success_rate,
                    avg_runtime_sec=excluded.avg_runtime_sec,
                    created_at=excluded.created_at
                """,
                (
                    experiment_name,
                    metadata.get("phase"),
                    metadata.get("model"),
                    metadata.get("sample_size"),
                    metadata.get("structure"),
                    int(bool(metadata.get("use_rag"))),
                    metadata.get("knowledge_base_mode"),
                    metadata.get("ablation"),
                    metadata.get("ensemble_method"),
                    json.dumps(metadata.get("weights")),
                    metadata.get("dataset_name"),
                    json.dumps(metrics),
                    metrics.get("json_parse_success_rate"),
                    metrics.get("avg_runtime_sec"),
                    now,
                ),
            )

    def save_agent_output(
        self,
        case_id: str,
        experiment_name: str,
        agent_name: str,
        output_json: dict | None,
        raw_output: str,
        evidence: list[str] | None = None,
        retrieval: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_outputs
                (case_id, experiment_name, agent_name, output_json, raw_output, evidence,
                 retrieval_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    experiment_name,
                    agent_name,
                    json.dumps(output_json) if output_json else None,
                    raw_output,
                    json.dumps(evidence or []),
                    json.dumps(retrieval) if retrieval else None,
                    now,
                ),
            )

    def save_prediction(
        self,
        case_id: str,
        experiment_name: str,
        predicted_risk: str,
        predicted_symptoms: list[str],
        final_response: str,
        hallucination_flag: bool = False,
        unsafe_flag: bool = False,
        run_metadata: dict[str, Any] | None = None,
        retrieval: dict[str, Any] | None = None,
    ) -> None:
        meta = run_metadata or {}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    case_id, experiment_name, predicted_risk, predicted_symptoms,
                    final_response, hallucination_flag, unsafe_flag,
                    model, sample_size, use_rag, structure, ablation,
                    ensemble_method, weights, phase, knowledge_base_mode, retrieval_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id, experiment_name) DO UPDATE SET
                    predicted_risk=excluded.predicted_risk,
                    predicted_symptoms=excluded.predicted_symptoms,
                    final_response=excluded.final_response,
                    hallucination_flag=excluded.hallucination_flag,
                    unsafe_flag=excluded.unsafe_flag,
                    model=excluded.model,
                    sample_size=excluded.sample_size,
                    use_rag=excluded.use_rag,
                    structure=excluded.structure,
                    ablation=excluded.ablation,
                    ensemble_method=excluded.ensemble_method,
                    weights=excluded.weights,
                    phase=excluded.phase,
                    knowledge_base_mode=excluded.knowledge_base_mode,
                    retrieval_json=excluded.retrieval_json
                """,
                (
                    case_id,
                    experiment_name,
                    predicted_risk,
                    json.dumps(predicted_symptoms),
                    final_response,
                    int(hallucination_flag),
                    int(unsafe_flag),
                    meta.get("model"),
                    meta.get("sample_size"),
                    int(bool(meta.get("use_rag"))),
                    meta.get("structure"),
                    meta.get("ablation"),
                    meta.get("ensemble_method"),
                    json.dumps(meta.get("weights")),
                    meta.get("phase"),
                    meta.get("knowledge_base_mode"),
                    json.dumps(retrieval) if retrieval else None,
                ),
            )

    def save_metric(self, experiment_name: str, metric_name: str, metric_value: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metrics (experiment_name, metric_name, metric_value)
                VALUES (?, ?, ?)
                ON CONFLICT(experiment_name, metric_name) DO UPDATE SET
                    metric_value=excluded.metric_value
                """,
                (experiment_name, metric_name, metric_value),
            )

    def count_agent_outputs(self, experiment_name: str | None = None) -> int:
        with self._connect() as conn:
            if experiment_name:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM agent_outputs WHERE experiment_name=?",
                    (experiment_name,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS c FROM agent_outputs").fetchone()
            return int(row["c"])

    def fetch_cases(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM cases").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["symptom_labels"] = json.loads(d.get("symptom_labels") or "[]")
            out.append(d)
        return out

    def fetch_case(self, case_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["symptom_labels"] = json.loads(d.get("symptom_labels") or "[]")
        return d

    def fetch_predictions(self, experiment_name: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if experiment_name:
                rows = conn.execute(
                    "SELECT * FROM predictions WHERE experiment_name=?",
                    (experiment_name,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM predictions").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["predicted_symptoms"] = json.loads(d.get("predicted_symptoms") or "[]")
            d["hallucination_flag"] = bool(d.get("hallucination_flag"))
            d["unsafe_flag"] = bool(d.get("unsafe_flag"))
            if d.get("retrieval_json"):
                try:
                    d["retrieval_json"] = json.loads(d["retrieval_json"])
                except json.JSONDecodeError:
                    pass
            out.append(d)
        return out

    def fetch_agent_outputs(
        self,
        case_id: str | None = None,
        experiment_name: str | None = None,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM agent_outputs WHERE 1=1"
        params: list[Any] = []
        if case_id:
            q += " AND case_id=?"
            params.append(case_id)
        if experiment_name:
            q += " AND experiment_name=?"
            params.append(experiment_name)
        q += " ORDER BY case_id, experiment_name, id"
        with self._connect() as conn:
            rows = conn.execute(q, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if d.get("output_json"):
                try:
                    d["output_json"] = json.loads(d["output_json"])
                except json.JSONDecodeError:
                    pass
            if d.get("evidence"):
                try:
                    d["evidence"] = json.loads(d["evidence"])
                except json.JSONDecodeError:
                    pass
            if d.get("retrieval_json"):
                try:
                    d["retrieval_json"] = json.loads(d["retrieval_json"])
                except json.JSONDecodeError:
                    pass
            out.append(d)
        return out

    def fetch_agent_trace(
        self, case_id: str, experiment_name: str
    ) -> list[dict[str, Any]]:
        return self.fetch_agent_outputs(case_id=case_id, experiment_name=experiment_name)

    def save_phase2_counsel_row(self, row: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO phase2_counsel_benchmark (
                    phase, model, structure, case_id, runtime_sec,
                    bertscore_f1, rouge_l, judge_faithfulness, judge_answer_relevancy,
                    judge_empathy, judge_safety, retrieval_relevance,
                    json_success_rate, token_throughput, inference_success,
                    experiment_key, metrics_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_key, case_id) DO UPDATE SET
                    runtime_sec=excluded.runtime_sec,
                    bertscore_f1=excluded.bertscore_f1,
                    rouge_l=excluded.rouge_l,
                    judge_faithfulness=excluded.judge_faithfulness,
                    judge_answer_relevancy=excluded.judge_answer_relevancy,
                    judge_empathy=excluded.judge_empathy,
                    judge_safety=excluded.judge_safety,
                    retrieval_relevance=excluded.retrieval_relevance,
                    json_success_rate=excluded.json_success_rate,
                    token_throughput=excluded.token_throughput,
                    inference_success=excluded.inference_success,
                    metrics_json=excluded.metrics_json,
                    created_at=excluded.created_at
                """,
                (
                    row.get("phase", "phase2"),
                    row["model"],
                    row["structure"],
                    row.get("case_id"),
                    row.get("runtime_sec"),
                    row.get("bertscore_f1"),
                    row.get("rouge_l"),
                    row.get("judge_faithfulness"),
                    row.get("judge_answer_relevancy"),
                    row.get("judge_empathy"),
                    row.get("judge_safety"),
                    row.get("retrieval_relevance"),
                    row.get("json_success_rate"),
                    row.get("token_throughput"),
                    int(bool(row.get("inference_success", True))),
                    row["experiment_key"],
                    json.dumps(row.get("metrics_json") or {}),
                    now,
                ),
            )

    def fetch_phase2_counsel_rows(
        self, experiment_key: str | None = None
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if experiment_key:
                rows = conn.execute(
                    "SELECT * FROM phase2_counsel_benchmark WHERE experiment_key=?",
                    (experiment_key,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM phase2_counsel_benchmark").fetchall()
        return [dict(r) for r in rows]
