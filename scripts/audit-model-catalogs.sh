#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if command -v python3 >/dev/null 2>&1; then
  python_bin=python3
elif command -v python >/dev/null 2>&1; then
  python_bin=python
else
  echo "[audit-model-catalogs] python3 or python is required" >&2
  exit 127
fi

"${python_bin}" .agents/skills/siumai-ai-sdk-maintenance/scripts/audit_model_catalogs.py \
  --include-green \
  --show-skipped \
  --defer deepinfra \
  "$@"
