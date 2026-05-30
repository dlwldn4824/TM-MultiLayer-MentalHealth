#!/usr/bin/env python3
"""Run all experiments (wrapper around src.main)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.main import main

if __name__ == "__main__":
    main()
