#!/usr/bin/env zsh
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/test-smoke.sh

Environment:
  SIUMAI_SMOKE_PROFILE        Optional preset for a provider feature set.
                             Values:
                               - openai (default)
                               - openai-compatible (openai + openai_compatible + groq + xai)
                               - all-providers
                               - custom (use the *_FEATURES vars below)

  SIUMAI_CORE_FEATURES        Optional features for crate `siumai-core`.
                             Default: depends on SIUMAI_SMOKE_PROFILE
  SIUMAI_PROVIDERS_FEATURES   Optional features for crate `siumai-providers`.
                             Default: depends on SIUMAI_SMOKE_PROFILE
  SIUMAI_REGISTRY_FEATURES    Optional features for crate `siumai-registry`.
                             Default: depends on SIUMAI_SMOKE_PROFILE

  SIUMAI_TEST_FACADE          Set to 1 to also run a small set of facade (`siumai`) tests.
                             Default: 0 (facade tests are slower to compile).
  SIUMAI_FEATURES             Features for crate `siumai` when SIUMAI_TEST_FACADE=1.
                             Default: depends on SIUMAI_SMOKE_PROFILE

Notes:
  - This script is a middle ground between `test-fast` and `test-full`.
  - It enables a minimal provider feature set so protocol-layer code paths compile and run.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

core_features="${SIUMAI_CORE_FEATURES:-openai}"
providers_features="${SIUMAI_PROVIDERS_FEATURES:-openai}"
registry_features="${SIUMAI_REGISTRY_FEATURES:-openai}"

profile="${SIUMAI_SMOKE_PROFILE:-openai}"
case "${profile}" in
  openai)
    : "${SIUMAI_CORE_FEATURES:=openai}"
    : "${SIUMAI_PROVIDERS_FEATURES:=openai}"
    : "${SIUMAI_REGISTRY_FEATURES:=openai}"
    : "${SIUMAI_FEATURES:=openai}"
    ;;
  openai-compatible)
    # Goal: keep a fast but meaningful refactor safety net for the OpenAI-compatible stack.
    # Note: `openai_compatible` provider module is behind the `openai` cargo feature.
    : "${SIUMAI_CORE_FEATURES:=openai,groq,xai}"
    : "${SIUMAI_PROVIDERS_FEATURES:=openai,groq,xai}"
    : "${SIUMAI_REGISTRY_FEATURES:=openai,groq,xai}"
    : "${SIUMAI_FEATURES:=openai,groq,xai}"
    ;;
  all-providers)
    : "${SIUMAI_CORE_FEATURES:=all-providers}"
    : "${SIUMAI_PROVIDERS_FEATURES:=all-providers}"
    : "${SIUMAI_REGISTRY_FEATURES:=all-providers}"
    : "${SIUMAI_FEATURES:=all-providers}"
    ;;
  custom)
    # Use the *_FEATURES vars as-is (with their defaults).
    ;;
  *)
    echo "[test-smoke] Unknown SIUMAI_SMOKE_PROFILE='${profile}'. Use -h for help."
    exit 2
    ;;
esac

core_features="${SIUMAI_CORE_FEATURES:-openai}"
providers_features="${SIUMAI_PROVIDERS_FEATURES:-openai}"
registry_features="${SIUMAI_REGISTRY_FEATURES:-openai}"

echo "[test-smoke] Running core unit tests (features: ${core_features})..."
cargo test -p siumai-core --lib --features "${core_features}"

echo "[test-smoke] Running providers unit tests (features: ${providers_features})..."
cargo test -p siumai-providers --lib --features "${providers_features}"

echo "[test-smoke] Running registry unit tests (features: ${registry_features})..."
cargo test -p siumai-registry --lib --features "${registry_features}"

if [[ "${SIUMAI_TEST_FACADE:-0}" == "1" ]]; then
  # Facade crate: optional, slower (dev-dependencies pull extra crates).
  # Run without default features (default enables all providers and is slow).
  # Note: siumai enforces "at least one provider feature" in build.rs.
  features="${SIUMAI_FEATURES:-openai}"
  echo "[test-smoke] Running facade tests (features: ${features})..."
  cargo test -p siumai --no-default-features --features "${features}" --lib
else
  echo "[test-smoke] Skipping crate siumai (set SIUMAI_TEST_FACADE=1 to include)"
fi

echo "[test-smoke] OK"
