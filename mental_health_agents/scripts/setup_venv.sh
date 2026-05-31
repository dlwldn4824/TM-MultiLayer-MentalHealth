#!/usr/bin/env bash
# Create .venv and install pilot dependencies (run once per machine).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  PY=python3
  command -v python3.9 >/dev/null && PY=python3.9
  "$PY" -m venv .venv
  echo "Created .venv ($PY)"
fi

.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
echo "Done. Run pilot with:"
echo "  python scripts/run_pilot.py --eval-only --no-rebuild-kb"
echo "  # or: .venv/bin/python scripts/run_pilot.py ..."
