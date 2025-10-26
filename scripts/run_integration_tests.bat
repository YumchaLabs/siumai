@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 Real LLM Integration Test Runner
echo ==================================
echo.

REM Load .env file if it exists
if exist ".env" (
    echo 📄 Loading environment variables from .env file...
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        REM Skip comments and empty lines
        echo %%a | findstr /r "^[^#]" >nul
        if not errorlevel 1 (
            set "%%a=%%b"
        )
    )
    echo ✅ .env file loaded
    echo.
) else (
    echo 💡 No .env file found. You can create one from .env.example
    echo.
)

set providers_configured=0
set total_providers=9

echo 📋 Checking environment variables...
echo.

REM Check OpenAI
if defined OPENAI_API_KEY (
    echo ✅ OPENAI_API_KEY is set
    set /a providers_configured+=1
) else (
    echo ❌ OPENAI_API_KEY is not set
    set /p "openai_key=🔑 Enter your OpenAI API key (or press Enter to skip): "
    if not "!openai_key!"=="" (
        set OPENAI_API_KEY=!openai_key!
        echo ✅ OPENAI_API_KEY set for this session
        set /a providers_configured+=1
    ) else (
        echo ⏭️ Skipping OpenAI
    )
)

REM Check Anthropic
if defined ANTHROPIC_API_KEY (
    echo ✅ ANTHROPIC_API_KEY is set
    set /a providers_configured+=1
) else (
    echo ❌ ANTHROPIC_API_KEY is not set
    set /p "anthropic_key=🔑 Enter your Anthropic API key (or press Enter to skip): "
    if not "!anthropic_key!"=="" (
        set ANTHROPIC_API_KEY=!anthropic_key!
        echo ✅ ANTHROPIC_API_KEY set for this session
        set /a providers_configured+=1
    ) else (
        echo ⏭️ Skipping Anthropic
    )
)

REM Check Gemini
if defined GEMINI_API_KEY (
    echo ✅ GEMINI_API_KEY is set
    set /a providers_configured+=1
) else (
    echo ❌ GEMINI_API_KEY is not set
    set /p "gemini_key=🔑 Enter your Google Gemini API key (or press Enter to skip): "
    if not "!gemini_key!"=="" (
        set GEMINI_API_KEY=!gemini_key!
        echo ✅ GEMINI_API_KEY set for this session
        set /a providers_configured+=1
    ) else (
        echo ⏭️ Skipping Gemini
    )
)

REM Check DeepSeek
if defined DEEPSEEK_API_KEY (
    echo ✅ DEEPSEEK_API_KEY is set
    set /a providers_configured+=1
) else (
    echo ❌ DEEPSEEK_API_KEY is not set
    set /p "deepseek_key=🔑 Enter your DeepSeek API key (or press Enter to skip): "
    if not "!deepseek_key!"=="" (
        set DEEPSEEK_API_KEY=!deepseek_key!
        echo ✅ DEEPSEEK_API_KEY set for this session
        set /a providers_configured+=1
    ) else (
        echo ⏭️ Skipping DeepSeek
    )
)

REM Check OpenRouter
if defined OPENROUTER_API_KEY (
    echo ✅ OPENROUTER_API_KEY is set
    set /a providers_configured+=1
) else (
    echo ❌ OPENROUTER_API_KEY is not set
    set /p "openrouter_key=🔑 Enter your OpenRouter API key (or press Enter to skip): "
    if not "!openrouter_key!"=="" (
        set OPENROUTER_API_KEY=!openrouter_key!
        echo ✅ OPENROUTER_API_KEY set for this session
        set /a providers_configured+=1
    ) else (
        echo ⏭️ Skipping OpenRouter
    )
)

REM Check Groq
if defined GROQ_API_KEY (
    echo ✅ GROQ_API_KEY is set
    set /a providers_configured+=1
) else (
    echo ❌ GROQ_API_KEY is not set
    set /p "groq_key=🔑 Enter your Groq API key (or press Enter to skip): "
    if not "!groq_key!"=="" (
        set GROQ_API_KEY=!groq_key!
        echo ✅ GROQ_API_KEY set for this session
        set /a providers_configured+=1
    ) else (
        echo ⏭️ Skipping Groq
    )
)

REM Check xAI
if defined XAI_API_KEY (
    echo ✅ XAI_API_KEY is set
    set /a providers_configured+=1
) else (
    echo ❌ XAI_API_KEY is not set
    set /p "xai_key=🔑 Enter your xAI API key (or press Enter to skip): "
    if not "!xai_key!"=="" (
        set XAI_API_KEY=!xai_key!
        echo ✅ XAI_API_KEY set for this session
        set /a providers_configured+=1
    ) else (
        echo ⏭️ Skipping xAI
)
)

REM Check SiliconFlow (for Rerank tests)
if defined SILICONFLOW_API_KEY (
    echo ✅ SILICONFLOW_API_KEY is set
    set /a providers_configured+=1
) else (
    echo ❌ SILICONFLOW_API_KEY is not set
    set /p "siliconflow_key=🔑 Enter your SiliconFlow API key (or press Enter to skip): "
    if not "!siliconflow_key!"=="" (
        set SILICONFLOW_API_KEY=!siliconflow_key!
        echo ✅ SILICONFLOW_API_KEY set for this session
        set /a providers_configured+=1
    ) else (
        echo ⏭️ Skipping SiliconFlow
    )
)

REM Check Ollama
echo.
echo 🦙 Checking Ollama availability...
if not defined OLLAMA_BASE_URL (
    set OLLAMA_BASE_URL=http://localhost:11434
)
curl -s "!OLLAMA_BASE_URL!/api/tags" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Ollama is available at !OLLAMA_BASE_URL!
    set /a providers_configured+=1
) else (
    echo ❌ Ollama is not available at !OLLAMA_BASE_URL!
    echo 💡 To enable Ollama tests:
    echo    1. Install Ollama: https://ollama.ai
    echo    2. Start Ollama: ollama serve
    echo    3. Pull models: ollama pull llama3.2:3b ^&^& ollama pull deepseek-r1:8b ^&^& ollama pull nomic-embed-text
)

echo.
echo 📊 Summary: !providers_configured!/!total_providers! providers configured
echo.

if !providers_configured! equ 0 (
    echo ❌ No providers configured. Please set at least one API key.
    echo.
    echo Example:
    echo set OPENAI_API_KEY=your-api-key
    echo set ANTHROPIC_API_KEY=your-api-key
    echo.
    pause
    exit /b 1
)

REM Check for optional base URL overrides
echo 🔧 Optional configuration:
if defined OPENAI_BASE_URL (
    echo    Using custom OpenAI base URL: %OPENAI_BASE_URL%
)
if defined ANTHROPIC_BASE_URL (
    echo    Using custom Anthropic base URL: %ANTHROPIC_BASE_URL%
)

echo.
echo 🧪 Running integration tests...
echo.

REM Ask user which test to run
echo Which test would you like to run?
echo 1^) All available providers - Basic tests ^(Chat, Streaming, Embedding, Reasoning^)
echo 2^) All capability tests - Tools, Vision, Audio, Image, Rerank
echo 3^) Specific capability test
echo 4^) Specific provider test
echo 5^) Provider interface tests ^(Provider::* vs Siumai::builder(^)^)
echo.
set /p "choice=Enter your choice (1-5): "

if "%choice%"=="1" (
    echo 🚀 Running basic provider tests...
    cargo test test_all_available_providers -- --ignored --nocapture
) else if "%choice%"=="2" (
    echo 🚀 Running comprehensive capability tests...
    echo.
    echo 📋 Running basic provider tests...
    cargo test test_all_available_providers -- --ignored --nocapture
    echo.
    echo 🔧 Running tool capability tests...
    cargo test test_all_provider_tools -- --ignored --nocapture
    echo.
    echo 👁️ Running vision capability tests...
    cargo test test_all_provider_vision -- --ignored --nocapture
    echo.
    echo 🔊 Running audio capability tests...
    cargo test test_all_provider_audio -- --ignored --nocapture
    echo.
    echo 🖼️ Running image integration tests (OpenAI)...
    cargo test test_openai_image_generation_integration -- --ignored --nocapture
    echo.
    echo 💎 Running image integration tests (Gemini)...
    cargo test test_gemini_image_generation_integration -- --ignored --nocapture
    echo.
    echo 🧮 Running rerank capability tests...
    REM Requires SILICONFLOW_API_KEY to exercise live rerank integration tests
    cargo test siliconflow_rerank_test -- --ignored --nocapture
) else if "%choice%"=="3" (
    echo.
    echo Available capability tests:
    echo - test_all_provider_tools ^(Tool calling across providers^)
    echo - test_all_provider_vision ^(Vision/multimodal^)
    echo - test_all_provider_audio ^(Audio TTS/STT^)
    echo - test_openai_image_generation_integration ^(OpenAI image integration^)
    echo - test_gemini_image_generation_integration ^(Gemini image integration^)
    echo - siliconflow_rerank_test ^(Rerank integration^)
    echo - test_all_available_providers ^(Basic smoke tests^)
    echo.
    set /p "test_name=Enter test name: "
    cargo test !test_name! -- --ignored --nocapture
) else if "%choice%"=="4" (
    echo.
    echo Available provider tests:
    echo - test_openai_integration
    echo - test_anthropic_integration
    echo - test_gemini_integration
    echo - test_deepseek_integration
    echo - test_openrouter_integration
    echo - test_groq_integration
    echo - test_xai_integration
    echo - test_ollama_integration
    echo.
    set /p "test_name=Enter test name: "
    cargo test !test_name! -- --ignored --nocapture
) else if "%choice%"=="5" (
    echo 🚀 Running provider interface tests...
    cargo test test_all_provider_interfaces -- --ignored --nocapture
) else (
    echo ❌ Invalid choice. Exiting.
    pause
    exit /b 1
)

echo.
echo ✅ Integration tests completed!
echo.
echo 💡 Tips:
echo - Tests automatically skip providers without API keys
echo - Some features may not be available for all API keys ^(this is normal^)
echo - Check the output for any warnings about missing permissions
echo.
pause
