"""SQLite database for evaluations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "auditor.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                model TEXT NOT NULL,
                structure TEXT NOT NULL,
                plain_response TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                plain_scores TEXT,
                agent_scores TEXT,
                runtime_plain REAL,
                runtime_agent REAL,
                pipeline_trace TEXT,
                insights TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS csv_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                filename TEXT NOT NULL,
                row_count INTEGER,
                summary_json TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    for key in ("plain_scores", "agent_scores", "pipeline_trace", "insights", "summary_json"):
        if key in d and isinstance(d[key], str) and d[key]:
            try:
                d[key] = json.loads(d[key])
            except json.JSONDecodeError:
                pass
    return d
