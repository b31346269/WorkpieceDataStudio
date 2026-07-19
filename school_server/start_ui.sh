#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The generation pipelines use logical cuda:0. Map it to one authorized
# physical card and hide every other card from the UI process.
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU:-2}"

PORT="${PORT:-7865}"
PID_FILE="$ROOT/workspace/school-ui.pid"
LOG_FILE="$ROOT/workspace/school-ui.log"
mkdir -p "$ROOT/workspace"

if [[ "${1:-}" == "--background" ]]; then
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Workpiece Data Studio is already running (PID $(cat "$PID_FILE"))."
    exit 0
  fi
  nohup .school-env/bin/python -m uvicorn workpiece_studio.main:app \
    --host 127.0.0.1 --port "$PORT" >"$LOG_FILE" 2>&1 </dev/null &
  echo "$!" >"$PID_FILE"
  echo "Started on remote 127.0.0.1:$PORT (PID $!)."
  exit 0
fi

exec .school-env/bin/python -m uvicorn workpiece_studio.main:app \
  --host 127.0.0.1 --port "$PORT"
