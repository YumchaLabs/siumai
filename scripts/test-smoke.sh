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
registry_features="${SIUMAI_REGISTRY_FEATURES:-openai}"

profile="${SIUMAI_SMOKE_PROFILE:-openai}"
case "${profile}" in
  openai)
    : "${SIUMAI_CORE_FEATURES:=openai}"
    : "${SIUMAI_REGISTRY_FEATURES:=openai}"
    : "${SIUMAI_FEATURES:=openai}"
    ;;
  openai-compatible)
    # Goal: keep a fast but meaningful refactor safety net for the OpenAI-compatible stack.
    # Note: `openai_compatible` provider module is behind the `openai` cargo feature.
    : "${SIUMAI_CORE_FEATURES:=openai,groq,xai}"
    : "${SIUMAI_REGISTRY_FEATURES:=openai,groq,xai}"
    : "${SIUMAI_FEATURES:=openai,groq,xai}"
    ;;
  all-providers)
    : "${SIUMAI_CORE_FEATURES:=all-providers}"
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
registry_features="${SIUMAI_REGISTRY_FEATURES:-openai}"

echo "[test-smoke] Running core unit tests (features: ${core_features})..."
if cargo nextest --version >/dev/null 2>&1; then
  nextest_profile="${SIUMAI_NEXTEST_PROFILE:-default}"
  echo "[test-smoke] Using nextest (profile: ${nextest_profile})"
  cargo nextest run --profile "${nextest_profile}" -p siumai-core --lib --features "${core_features}"
else
  cargo test -p siumai-core --lib --features "${core_features}"
fi

echo "[test-smoke] Running provider crate unit tests (registry features: ${registry_features})..."

run_provider_tests() {
  local crate="$1"
  local features="$2"
  if cargo nextest --version >/dev/null 2>&1; then
    local nextest_profile="${SIUMAI_NEXTEST_PROFILE:-default}"
    cargo nextest run --profile "${nextest_profile}" -p "${crate}" --lib --features "${features}"
  else
    cargo test -p "${crate}" --lib --features "${features}"
  fi
}

if [[ ",${registry_features}," == *",all-providers,"* ]]; then
  run_provider_tests siumai-provider-openai openai
  run_provider_tests siumai-provider-anthropic anthropic
  run_provider_tests siumai-provider-gemini google
  run_provider_tests siumai-provider-ollama ollama
  run_provider_tests siumai-provider-groq groq
  run_provider_tests siumai-provider-xai xai
  run_provider_tests siumai-provider-minimaxi minimaxi
else
  [[ ",${registry_features}," == *",openai,"* ]] && run_provider_tests siumai-provider-openai openai
  [[ ",${registry_features}," == *",anthropic,"* ]] && run_provider_tests siumai-provider-anthropic anthropic
  [[ ",${registry_features}," == *",google,"* ]] && run_provider_tests siumai-provider-gemini google
  [[ ",${registry_features}," == *",ollama,"* ]] && run_provider_tests siumai-provider-ollama ollama
  [[ ",${registry_features}," == *",groq,"* ]] && run_provider_tests siumai-provider-groq groq
  [[ ",${registry_features}," == *",xai,"* ]] && run_provider_tests siumai-provider-xai xai
  [[ ",${registry_features}," == *",minimaxi,"* ]] && run_provider_tests siumai-provider-minimaxi minimaxi
fi

echo "[test-smoke] Running registry unit tests (features: ${registry_features})..."
if cargo nextest --version >/dev/null 2>&1; then
  nextest_profile="${SIUMAI_NEXTEST_PROFILE:-default}"
  cargo nextest run --profile "${nextest_profile}" -p siumai-registry --lib --features "${registry_features}"
else
  cargo test -p siumai-registry --lib --features "${registry_features}"
fi

if [[ "${SIUMAI_TEST_FACADE:-0}" == "1" ]]; then
  # Facade crate: optional, slower (dev-dependencies pull extra crates).
  # Run without default features (default enables all providers and is slow).
  # Note: siumai enforces "at least one provider feature" in build.rs.
  features="${SIUMAI_FEATURES:-openai}"
  echo "[test-smoke] Running facade tests (features: ${features})..."
  if cargo nextest --version >/dev/null 2>&1; then
    nextest_profile="${SIUMAI_NEXTEST_PROFILE:-default}"
    cargo nextest run --profile "${nextest_profile}" -p siumai --no-default-features --features "${features}" --lib
  else
    cargo test -p siumai --no-default-features --features "${features}" --lib
  fi
else
  echo "[test-smoke] Skipping crate siumai (set SIUMAI_TEST_FACADE=1 to include)"
fi

echo "[test-smoke] OK"
