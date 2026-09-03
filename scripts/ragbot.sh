#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python was not found. Ragbot requires Python 3.10+." >&2
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/ragbot.py" "$@"
