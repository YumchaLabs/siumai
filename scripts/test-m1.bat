@echo off
setlocal enabledelayedexpansion

REM M1 smoke matrix for Alpha.5 (Core Trio: OpenAI / Anthropic / Gemini).
REM This is intended to be fast and focused:
REM - fixture inventory validation
REM - core transcoding tests (gateway/proxy)
REM - tool-loop gateway tests (siumai-extras)

cd /d "%~dp0\.." || exit /b 1

echo [m1] validate fixture inventory...
python -B scripts\check_fixture_inventory.py || exit /b 1

echo [m1] nextest: siumai core transcoding suite...
cargo nextest run -p siumai --tests --features "openai,anthropic,google" ^
  --test transcoding_anthropic_to_gemini_alignment_test ^
  --test transcoding_anthropic_to_openai_alignment_test ^
  --test transcoding_gemini_to_anthropic_alignment_test ^
  --test transcoding_gemini_to_openai_alignment_test ^
  --test transcoding_openai_to_openai_chat_completions_tool_approval_policy_test ^
  --test transcoding_openai_to_anthropic_alignment_test ^
  --test transcoding_openai_to_gemini_alignment_test ^
  --test gemini_function_response_gateway_roundtrip_test || exit /b 1

echo [m1] nextest: siumai-extras gateway/tool-loop suite...
cargo nextest run -p siumai-extras --tests --features "server,openai,google,anthropic" || exit /b 1

echo [m1] OK
exit /b 0
