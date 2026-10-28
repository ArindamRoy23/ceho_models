#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="${PARAKEET_REPO:-nvidia/parakeet-tdt-0.6b-v3}"
FILE="${PARAKEET_FILE:-parakeet-tdt-0.6b-v3.nemo}"
DEST="${PARAKEET_DIR:-${ROOT}/data/transcription/models}"

mkdir -p "$DEST"

exec uv run --directory "$ROOT" --package transcription hf download \
  "$REPO" \
  "$FILE" \
  --local-dir "$DEST" \
  "$@"
