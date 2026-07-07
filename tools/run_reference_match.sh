#!/usr/bin/env bash
# Reference runner (P0.2 ground truth).
#
# Runs a full 8-player / 1400-round match through the REAL pipe runner with the
# vendored engine, then refreshes the recorded fixtures. This is the ground
# truth the in-process training wrapper is proven equivalent to at P1.3.
#
# Usage:  tools/run_reference_match.sh [--refresh-fixtures]
#
# The match runs in real time (~140s: the engine sleeps to hold 0.1s/round).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv-bot/bin/python"
SIM="$REPO/.venv-bot/bin/simulation"
WS="$REPO/.agario/simulation"

cd "$REPO"
rm -rf "$WS"
"$SIM" --headless \
  4:tests/fixtures/bots/cautious.py \
  4:tests/fixtures/bots/dont_move.py

echo "[reference] match complete. Rankings:"
"$PY" -m json.tool "$WS/output/results.json"

if [[ "${1:-}" == "--refresh-fixtures" ]]; then
  cp "$WS/output/game.json" tests/fixtures/game.json
  cp "$WS/output/results.json" tests/fixtures/results.json
  echo "[reference] fixtures refreshed in tests/fixtures/"
fi
