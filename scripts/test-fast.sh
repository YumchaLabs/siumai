#!/usr/bin/env zsh
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/test-fast.sh

Environment:
  SIUMAI_TEST_FACADE  Set to 1 to also run crate `siumai` tests (slower; pulls heavier dev-deps).
  SIUMAI_FEATURES   Optional comma-separated feature list for crate `siumai`.
                   Example: SIUMAI_FEATURES="openai,google" ./scripts/test-fast.sh
  SIUMAI_PROVIDERS_FEATURES  Optional comma-separated feature list for crate `siumai-providers`.
                   Example: SIUMAI_PROVIDERS_FEATURES="openai,google" ./scripts/test-fast.sh
  SIUMAI_REGISTRY_FEATURES  Optional comma-separated feature list for crate `siumai-registry`.
                   Example: SIUMAI_REGISTRY_FEATURES="openai,google" ./scripts/test-fast.sh

Notes:
  - This script is optimized for fast iteration during refactors.
  - It intentionally avoids `siumai`'s default `all-providers` feature to reduce compile time.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

echo "[test-fast] Running core unit tests (no provider feature matrix)..."

# Core crates: fastest "refactor safety net"
cargo test -p siumai-core --lib

registry_features="${SIUMAI_REGISTRY_FEATURES:-}"
if [[ -n "${registry_features}" ]]; then
  echo "[test-fast] Testing crate siumai-registry with features: ${registry_features}"
  cargo test -p siumai-registry --lib --features "${registry_features}"
else
  cargo test -p siumai-registry --lib
fi

providers_features="${SIUMAI_PROVIDERS_FEATURES:-}"
if [[ -n "${providers_features}" ]]; then
  echo "[test-fast] Testing crate siumai-providers with features: ${providers_features}"
  cargo test -p siumai-providers --lib --features "${providers_features}"
else
  cargo test -p siumai-providers --lib
fi

if [[ "${SIUMAI_TEST_FACADE:-0}" == "1" ]]; then
  # Facade crate: optional, slower (dev-dependencies pull extra crates).
  # Run without default features (default enables all providers and is slow).
  # Note: siumai enforces "at least one provider feature" in build.rs.
  features="${SIUMAI_FEATURES:-openai}"
  echo "[test-fast] Testing crate siumai with features: ${features}"
  cargo test -p siumai --lib --no-default-features --features "${features}"
else
  echo "[test-fast] Skipping crate siumai (set SIUMAI_TEST_FACADE=1 to include)"
fi

echo "[test-fast] OK"
