#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/test-full.sh

Notes:
  - This script aims to match CI behavior (full workspace, all features).
  - If `cargo nextest` is installed, it will be used automatically.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if cargo nextest --version >/dev/null 2>&1; then
  echo "[test-full] Using nextest"
  cargo nextest run --profile ci --workspace --all-features
else
  echo "[test-full] nextest not found; falling back to cargo test"
  cargo test --workspace --all-features
fi

echo "[test-full] OK"
