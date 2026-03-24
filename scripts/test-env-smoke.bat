@echo off
setlocal enabledelayedexpansion

set "FEATURES=%SIUMAI_ENV_SMOKE_FEATURES%"
if "%FEATURES%"=="" set "FEATURES=openai,anthropic,google,deepseek,groq"

if not "%SIUMAI_TEST_PROXY%"=="" (
    set "HTTP_PROXY=%SIUMAI_TEST_PROXY%"
    set "HTTPS_PROXY=%SIUMAI_TEST_PROXY%"
    set "ALL_PROXY=%SIUMAI_TEST_PROXY%"
)

echo [test-env-smoke] features: %FEATURES%
if not "%HTTP_PROXY%"=="" (
    echo [test-env-smoke] proxy: %HTTP_PROXY%
)

cargo test ^
  -p siumai ^
  --test provider_env_smoke_test ^
  --no-default-features ^
  --features "%FEATURES%" ^
  -- ^
  --ignored ^
  --nocapture ^
  --test-threads=1
