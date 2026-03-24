#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/test-env-smoke.sh

Environment:
  SIUMAI_ENV_SMOKE_FEATURES   Cargo features for crate `siumai`.
                             Default: openai,anthropic,google,deepseek,groq

  SIUMAI_TEST_PROXY          Optional proxy URL. When set, the script exports:
                             HTTP_PROXY / HTTPS_PROXY / ALL_PROXY

  OPENAI_API_KEY             Optional. Enables OpenAI live smoke.
  OPENAI_BASE_URL            Optional. Also exercises explicit OpenAI base_url.
  OPENAI_MODEL               Optional. Override OpenAI smoke model.

  ANTHROPIC_API_KEY          Optional. Enables Anthropic live smoke.
  ANTHROPIC_BASE_URL         Optional. Also exercises explicit Anthropic base_url.
  ANTHROPIC_MODEL            Optional. Override Anthropic smoke model.

  GEMINI_API_KEY             Optional. Enables Gemini live smoke.
  GEMINI_MODEL               Optional. Override Gemini smoke model.

  DEEPSEEK_API_KEY           Optional. Enables DeepSeek live smoke.
  DEEPSEEK_MODEL             Optional. Override DeepSeek smoke model.

  GROQ_API_KEY               Optional. Enables Groq live smoke.
  GROQ_MODEL                 Optional. Override Groq smoke model.

Notes:
  - Tests are ignored by default and make real API calls.
  - If a provider API key is absent, the corresponding test self-skips.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -n "${SIUMAI_TEST_PROXY:-}" ]]; then
  export HTTP_PROXY="${SIUMAI_TEST_PROXY}"
  export HTTPS_PROXY="${SIUMAI_TEST_PROXY}"
  export ALL_PROXY="${SIUMAI_TEST_PROXY}"
fi

features="${SIUMAI_ENV_SMOKE_FEATURES:-openai,anthropic,google,deepseek,groq}"

echo "[test-env-smoke] features: ${features}"
if [[ -n "${HTTP_PROXY:-}" ]]; then
  echo "[test-env-smoke] proxy: ${HTTP_PROXY}"
fi

cargo test \
  -p siumai \
  --test provider_env_smoke_test \
  --no-default-features \
  --features "${features}" \
  -- \
  --ignored \
  --nocapture \
  --test-threads=1
