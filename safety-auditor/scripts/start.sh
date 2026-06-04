#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Starting backend on :8000..."
cd "$ROOT/backend"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACK_PID=$!

echo "Starting frontend on :3000..."
cd "$ROOT/frontend"
if [ ! -d node_modules ]; then
  npm install
fi
npm run dev &
FRONT_PID=$!

echo "Backend PID=$BACK_PID Frontend PID=$FRONT_PID"
echo "Open http://localhost:3000"
wait
