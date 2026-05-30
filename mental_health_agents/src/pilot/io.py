"""CSV checkpoint helpers (pandas NaN-safe JSON fields)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

DEBUG_LOG = PROJECT_ROOT.parent / ".cursor" / "debug-4e6bbd.log"


def _dbg(hypothesis_id: str, location: str, message: str, data: dict[str, Any] | None = None) -> None:
    #region agent log
    try:
        payload = {
            "sessionId": "4e6bbd",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    #endregion


def is_blank_json_cell(val: Any) -> bool:
    """True for None, NaN, empty string — pandas reads empty CSV cells as float NaN."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and not val.strip():
        return True
    return False


def parse_optional_json(val: Any, *, ctx: str = "") -> Any | None:
    if is_blank_json_cell(val):
        return None
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            _dbg("H2", "io.py:parse_optional_json", "json decode error", {"ctx": ctx, "prefix": val[:80]})
            return None
    _dbg(
        "H1",
        "io.py:parse_optional_json",
        "unexpected type passed to json parse",
        {"ctx": ctx, "type": type(val).__name__},
    )
    return None


def attach_retrieval_from_row(row: dict[str, Any]) -> None:
    if row.get("retrieval"):
        return
    parsed = parse_optional_json(row.get("retrieval_json"), ctx=f"case={row.get('case_id')}")
    row["retrieval"] = parsed
