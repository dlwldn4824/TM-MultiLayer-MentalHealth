#!/usr/bin/env python3
"""Phase 2: CounselChat model comparison benchmark."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_VENV_PY = ROOT / ".venv" / "bin" / "python"
_VENV_ROOT = (ROOT / ".venv").resolve()
if _VENV_PY.is_file() and Path(sys.prefix).resolve() != _VENV_ROOT:
    script = Path(__file__).resolve()
    os.execv(_VENV_PY, [str(_VENV_PY), str(script), *sys.argv[1:]])

from src.phase2_runner import main

if __name__ == "__main__":
    main()
