#!/usr/bin/env bash
# CounselChat pilot — checkpoint after every case; safe to interrupt and re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="outputs/pilot/pilot_run.log"
mkdir -p outputs/pilot

if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
  echo "Ollama is not running. Start it first: ollama serve"
  exit 1
fi

if pgrep -f "scripts/run_pilot.py" >/dev/null 2>&1; then
  echo "Pilot already running. Monitor:"
  echo "  tail -f $LOG"
  echo "  cat outputs/pilot/progress.json"
  exit 0
fi

echo "Starting pilot (resume=ON by default). Log: $LOG"
# Logs go to $LOG via FileHandler in run_pilot (flush each line for tail -f)
nohup .venv/bin/python scripts/run_pilot.py --no-rebuild-kb >> "$LOG" 2>&1 &
echo "PID=$!"
echo "Progress: outputs/pilot/progress.json"
echo "To resume after interrupt, run this script again."
