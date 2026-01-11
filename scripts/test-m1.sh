#!/usr/bin/env bash
set -euo pipefail

# M1 smoke matrix for Alpha.5 (Core Trio: OpenAI / Anthropic / Gemini).
# This is intended to be fast and focused:
# - fixture drift audit (repo-ref/ai vs siumai)
# - core transcoding tests (gateway/proxy)
# - tool-loop gateway tests (siumai-extras)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[m1] audit vercel fixtures..."
python scripts/audit_vercel_fixtures.py --ai-root ../ai --siumai-root .

echo "[m1] nextest: siumai core transcoding suite..."
cargo nextest run -p siumai --tests --features "openai,anthropic,google" \
  --test transcoding_anthropic_to_gemini_alignment_test \
  --test transcoding_anthropic_to_openai_alignment_test \
  --test transcoding_gemini_to_anthropic_alignment_test \
  --test transcoding_gemini_to_openai_alignment_test \
  --test transcoding_openai_to_openai_chat_completions_tool_approval_policy_test \
  --test transcoding_openai_to_anthropic_alignment_test \
  --test transcoding_openai_to_gemini_alignment_test \
  --test gemini_function_response_gateway_roundtrip_test

echo "[m1] nextest: siumai-extras gateway/tool-loop suite..."
cargo nextest run -p siumai-extras --tests --features "server,openai,google,anthropic"

echo "[m1] OK"
