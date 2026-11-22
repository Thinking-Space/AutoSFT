#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${AUTO_SFT_PYTHON:-python}" -m autosft run "$@"
