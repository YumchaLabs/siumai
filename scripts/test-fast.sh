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
  SIUMAI_REGISTRY_FEATURES  Optional comma-separated feature list for crate `siumai-registry`.
                   Example: SIUMAI_REGISTRY_FEATURES="openai,google" ./scripts/test-fast.sh
  SIUMAI_PROVIDER_PROFILE   Optional provider-crate test preset.
                   Values: openai (default) | openai-compatible | all-providers

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

provider_profile="${SIUMAI_PROVIDER_PROFILE:-openai}"
echo "[test-fast] Testing provider crates (profile: ${provider_profile})"
case "${provider_profile}" in
  openai)
    cargo test -p siumai-provider-openai --lib --features openai
    ;;
  openai-compatible)
    cargo test -p siumai-provider-openai --lib --features openai
    cargo test -p siumai-provider-groq --lib --features groq
    cargo test -p siumai-provider-xai --lib --features xai
    ;;
  all-providers)
    cargo test -p siumai-provider-openai --lib --features openai
    cargo test -p siumai-provider-anthropic --lib --features anthropic
    cargo test -p siumai-provider-gemini --lib --features google
    cargo test -p siumai-provider-ollama --lib --features ollama
    cargo test -p siumai-provider-groq --lib --features groq
    cargo test -p siumai-provider-xai --lib --features xai
    cargo test -p siumai-provider-minimaxi --lib --features minimaxi
    ;;
  *)
    echo "[test-fast] Unknown SIUMAI_PROVIDER_PROFILE='${provider_profile}'."
    exit 2
    ;;
esac

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
