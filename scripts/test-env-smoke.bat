@echo off
setlocal enabledelayedexpansion

if "%~1"=="-h" goto :usage
if "%~1"=="--help" goto :usage

set "PROFILE=%SIUMAI_ENV_SMOKE_PROFILE%"
if "%PROFILE%"=="" set "PROFILE=core-default"

set "FEATURES=%SIUMAI_ENV_SMOKE_FEATURES%"
if "%FEATURES%"=="" (
    if /I "%PROFILE%"=="core-default" (
        set "FEATURES=openai,anthropic,deepseek"
    ) else if /I "%PROFILE%"=="openai" (
        set "FEATURES=openai"
    ) else if /I "%PROFILE%"=="all-providers" (
        set "FEATURES=openai,anthropic,google,deepseek,groq"
    ) else if /I "%PROFILE%"=="custom" (
        echo [test-env-smoke] custom profile requires SIUMAI_ENV_SMOKE_FEATURES
        exit /b 2
    ) else (
        echo [test-env-smoke] Unknown SIUMAI_ENV_SMOKE_PROFILE="%PROFILE%". Use --help for help.
        exit /b 2
    )
)

if not "%SIUMAI_TEST_PROXY%"=="" (
    set "HTTP_PROXY=%SIUMAI_TEST_PROXY%"
    set "HTTPS_PROXY=%SIUMAI_TEST_PROXY%"
    set "ALL_PROXY=%SIUMAI_TEST_PROXY%"
)

echo [test-env-smoke] profile: %PROFILE%
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
exit /b %ERRORLEVEL%

:usage
echo Usage:
echo   scripts\test-env-smoke.bat
echo.
echo Environment:
echo   SIUMAI_ENV_SMOKE_PROFILE  Optional preset for the live provider set.
echo                            Values: core-default ^(default^), openai, all-providers, custom
echo   SIUMAI_ENV_SMOKE_FEATURES Optional custom feature list for crate siumai.
echo                            Used directly when PROFILE=custom.
echo   SIUMAI_TEST_PROXY         Optional proxy URL mirrored into HTTP_PROXY/HTTPS_PROXY/ALL_PROXY.
echo   SIUMAI_ENV_SMOKE_STRICT   Optional strict mode. Set to 1/true/yes/on to
echo                            fail instead of self-skipping known access denials.
echo.
echo Notes:
echo   - Tests are ignored by default and make real API calls.
echo   - If a provider API key is absent, the corresponding test self-skips.
echo   - core-default avoids providers commonly blocked by account or region.
echo   - Gemini/Groq also self-skip on known region/account access denials unless
echo     SIUMAI_ENV_SMOKE_STRICT is enabled.
exit /b 0
