#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/test-cross-feature-contracts.sh [profile]

Profiles:
  all                 Run all cross-feature contract profiles (default)
  openai-websocket    OpenAI websocket contract bundle
  google-gcp          Google/GCP contract bundle
  openai-json-repair  OpenAI JSON repair contract bundle

Notes:
  - These are no-network contract tests for multi-feature facade combinations.
  - `cargo nextest` is preferred. If unavailable, the script falls back to `cargo test`.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

profiles=(
  openai-websocket
  google-gcp
  openai-json-repair
)

selected_profile="${1:-all}"

run_contract_profile() {
  local profile="$1"
  local features=""
  local tests=()

  case "${profile}" in
    openai-websocket)
      features="openai,openai-websocket"
      tests=(
        openai_websocket_builder_smoke_test
        openai_unified_builder_api_mode_routing_test
      )
      ;;
    google-gcp)
      features="google,gcp"
      tests=(
        service_account_provider_test
        google_vertex_builder_alignment_test
        vertex_token_provider_alias_test
      )
      ;;
    openai-json-repair)
      features="openai,json-repair"
      tests=(
        json_repair_non_streaming_test
        structured_output_public_facade_test
      )
      ;;
    *)
      echo "[test-cross-feature-contracts] Unknown profile: ${profile}" >&2
      exit 2
      ;;
  esac

  local test_args=()
  local test_name=""
  for test_name in "${tests[@]}"; do
    test_args+=(--test "${test_name}")
  done

  echo "[test-cross-feature-contracts] profile=${profile} features=${features}"
  if cargo nextest --version >/dev/null 2>&1; then
    cargo nextest run -p siumai --no-default-features --features "${features}" --no-fail-fast "${test_args[@]}"
  else
    cargo test -p siumai --no-default-features --features "${features}" "${test_args[@]}"
  fi
}

if [[ "${selected_profile}" == "all" ]]; then
  for profile in "${profiles[@]}"; do
    run_contract_profile "${profile}"
  done
else
  run_contract_profile "${selected_profile}"
fi

echo "[test-cross-feature-contracts] OK"
