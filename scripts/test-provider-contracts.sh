#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/test-provider-contracts.sh [profile]

Profiles:
  all             Run all provider contract profiles (default)
  openai-native   OpenAI native facade contract tests
  openai-compat   OpenAI-compatible vendor/preset contract tests
  azure           Azure facade contract tests
  anthropic       Anthropic facade contract tests
  google          Gemini / Google facade contract tests
  google-vertex   Google Vertex + Anthropic-on-Vertex facade contract tests
  ollama          Ollama facade contract tests
  xai             xAI facade contract tests
  groq            Groq facade contract tests
  minimaxi        MiniMaxi facade contract tests
  deepseek        DeepSeek facade contract tests
  cohere          Cohere facade contract tests
  togetherai      TogetherAI facade contract tests
  bedrock         Amazon Bedrock facade contract tests

Notes:
  - These are provider-scoped, no-network contract tests for crate `siumai`.
  - `cargo nextest` is preferred. If unavailable, the script falls back to `cargo test`.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

profiles=(
  openai-native
  openai-compat
  azure
  anthropic
  google
  google-vertex
  ollama
  xai
  groq
  minimaxi
  deepseek
  cohere
  togetherai
  bedrock
)

selected_profile="${1:-all}"

run_contract_profile() {
  local profile="$1"
  local features=""
  local tests=()

  case "${profile}" in
    openai-native)
      features="openai"
      tests=(
        openai_chat_custom_transport_alignment_test
        openai_responses_text_stream_alignment_test
        openai_unified_builder_api_mode_routing_test
      )
      ;;
    openai-compat)
      features="openai"
      tests=(
        openai_compatible_preset_guards_test
        openrouter_chat_request_alignment_test
        perplexity_chat_request_alignment_test
      )
      ;;
    azure)
      features="azure"
      tests=(
        azure_chat_custom_transport_alignment_test
        azure_openai_provider_request_fixtures_alignment_test
      )
      ;;
    anthropic)
      features="anthropic"
      tests=(
        anthropic_messages_fixtures_alignment_test
        anthropic_messages_stream_fixtures_alignment_test
      )
      ;;
    google)
      features="google"
      tests=(
        google_generative_ai_fixtures_alignment_test
        google_generative_ai_stream_fixtures_alignment_test
        gemini_builder_common_params_test
      )
      ;;
    google-vertex)
      features="google-vertex,gcp"
      tests=(
        vertex_chat_fixtures_alignment_test
        anthropic_vertex_builder_alignment_test
        vertex_token_provider_alias_test
      )
      ;;
    ollama)
      features="ollama"
      tests=(
        ollama_chat_request_fixtures_alignment_test
        ollama_http_error_fixtures_alignment_test
      )
      ;;
    xai)
      features="xai"
      tests=(
        xai_chat_request_fixtures_alignment_test
        xai_responses_response_fixtures_alignment_test
        xai_responses_text_stream_alignment_test
      )
      ;;
    groq)
      features="groq"
      tests=(
        groq_chat_request_fixtures_alignment_test
        groq_chat_custom_transport_alignment_test
      )
      ;;
    minimaxi)
      features="minimaxi"
      tests=(
        minimaxi_chat_request_fixtures_alignment_test
        minimaxi_http_error_fixtures_alignment_test
      )
      ;;
    deepseek)
      features="deepseek"
      tests=(
        deepseek_chat_response_alignment_test
        deepseek_chat_stream_alignment_test
      )
      ;;
    cohere)
      features="cohere"
      tests=(
        cohere_rerank_fixtures_alignment_test
        cohere_http_error_fixtures_alignment_test
      )
      ;;
    togetherai)
      features="togetherai"
      tests=(
        togetherai_rerank_fixtures_alignment_test
        togetherai_http_error_fixtures_alignment_test
      )
      ;;
    bedrock)
      features="bedrock"
      tests=(
        bedrock_chat_request_fixtures_alignment_test
        bedrock_rerank_response_alignment_test
        bedrock_chat_stream_alignment_test
      )
      ;;
    *)
      echo "[test-provider-contracts] Unknown profile: ${profile}" >&2
      exit 2
      ;;
  esac

  local test_args=()
  local test_name=""
  for test_name in "${tests[@]}"; do
    test_args+=(--test "${test_name}")
  done

  echo "[test-provider-contracts] profile=${profile} features=${features}"
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

echo "[test-provider-contracts] OK"
